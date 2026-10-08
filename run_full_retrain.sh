#!/usr/bin/env bash
#
# Retreino completo das 5 regioes do Brasil com a formulacao
# persistencia+delta (SPI defasado como canal de entrada extra, modelo
# preve SPI = ultimo SPI observado + correcao aprendida - ver
# models/spi_predictor.py e data/dataset.py). Substitui os checkpoints e
# resultados de grid search usados no paper atual.
#
# Loss com peso de severidade (training/losses.py::MaskedMSELoss): mesmo
# apos a ancoragem por persistencia, a correcao aprendida ainda e treinada
# com MSE simples, que empurra a previsao para a media condicional sempre
# que o sinal climatico nao resolve totalmente a anomalia - o mesmo
# mecanismo de atenuacao ja diagnosticado para a formulacao em valor
# absoluto, agindo agora sobre a correcao em vez do SPI bruto. Isso aparece
# nos mapas de classificacao do paper (Nordeste, Centro-Oeste e Sudeste com
# area seca prevista bem menor que a observada). SEVERITY_WEIGHT>0 pondera
# o erro quadratico de cada mes por (1 + SEVERITY_WEIGHT * |SPI observado|),
# dando mais peso a meses mais extremos no gradiente; SEVERITY_WEIGHT=0
# reproduz exatamente o MSE simples. O padrao 0.5 e o valor usado no paper
# (lambda=0.5). Exporte a variavel antes de chamar o script para testar
# outros pesos, ex.: SEVERITY_WEIGHT=2.0 ./run_full_retrain.sh
SEVERITY_WEIGHT="${SEVERITY_WEIGHT:-0.5}"
#
# REUSE_EXISTING=1: pula precompute-spi e train-ae para uma regiao quando o
# cache de SPI (outputs/<regiao>/spi_cache/spi_scale_3.pkl) ou o checkpoint
# do autoencoder (outputs/<regiao>/autoencoder/model.pth) ja existirem no
# disco - nenhum dos dois depende de SEVERITY_WEIGHT (precompute-spi so usa
# a precipitacao; train-ae usa uma loss de reconstrucao completamente
# separada, WeightedSmoothL1Loss, que nunca ve o SPI predito), entao os
# artefatos de um retreino anterior continuam validos para gerar um novo
# grid-search com peso de severidade diferente. Uma regiao sem cache/
# checkpoint existente roda normalmente mesmo com REUSE_EXISTING=1 (o skip
# e por regiao, nao tudo-ou-nada). grid-search NUNCA e pulado, pois e
# exatamente o que muda com SEVERITY_WEIGHT.
# Padrao: 0 (sempre roda tudo do zero, comportamento original do script).
# Uso: REUSE_EXISTING=1 ./run_full_retrain.sh
REUSE_EXISTING="${REUSE_EXISTING:-0}"
#
# Para cada regiao, roda em sequencia (1 e 2 pulados por regiao se
# REUSE_EXISTING=1 e o artefato ja existir - ver run_step_reusable):
#   1) precompute-spi   - recalcula o cache de SPI (idempotente; sempre
#                          seguro rodar de novo, mesmo algoritmo/dados)
#   2) train-ae          - pretreina o autoencoder (sempre retreina do
#                          zero - ver experiments/train_autoencoder.py -
#                          agora com o canal SPI incluido)
#   3) grid-search --use-transfer-learning   - grade p x q completa, TL
#   4) grid-search (scratch)                 - grade p x q completa, scratch
#
# Uso:
#   ./run_full_retrain.sh                       # todas as 5 regioes, do zero
#   ./run_full_retrain.sh Sul Nordeste           # so as regioes passadas
#   REUSE_EXISTING=1 ./run_full_retrain.sh       # reaproveita SPI/AE existentes,
#                                                 # so roda grid-search de novo
#
# Rode a partir da raiz do pacote (onde ficam main.py e a pasta config/),
# ex.: drought_forecast_regression/
#
# Se algum comando falhar, o script registra o erro em failed_runs.log e
# continua com as demais etapas (nao usa 'set -e').
#
# Pre-requisitos no servidor:
#   - pasta de rasters brutos disponivel (ajustar DATA_BASE_PATH em
#     config/paths.py se o caminho for diferente do local)
#   - pip install -r requirements.txt + PyTorch com CUDA (o requirements.txt
#     nao fixa uma build CUDA especifica - instalar separadamente conforme
#     o driver/CUDA do servidor)

set -uo pipefail

REGIONS=("$@")
if [ ${#REGIONS[@]} -eq 0 ]; then
    REGIONS=("Sul" "Sudeste" "Nordeste" "Centro-Oeste" "Norte")
fi

echo "Peso de severidade da loss (SEVERITY_WEIGHT): $SEVERITY_WEIGHT"
echo "Reaproveitar SPI/autoencoder existentes (REUSE_EXISTING): $REUSE_EXISTING"

LOG_FILE="run_full_retrain_$(date +%Y%m%d_%H%M%S).log"
FAILED_LOG="failed_retrain_runs.log"
: > "$FAILED_LOG"

run_step() {
    local desc="$1"
    shift
    local t0 t1
    t0=$(date +%s)
    echo ">>> ${desc}" | tee -a "$LOG_FILE"
    if "$@" >>"$LOG_FILE" 2>&1; then
        t1=$(date +%s)
        echo "    OK (${desc}) em $((t1 - t0))s" | tee -a "$LOG_FILE"
    else
        t1=$(date +%s)
        echo "FALHOU: $desc (apos $((t1 - t0))s)" | tee -a "$FAILED_LOG"
        echo "    FALHOU (${desc}) - veja $LOG_FILE" | tee -a "$LOG_FILE"
    fi
}

# Roda $2.. (o comando) a menos que REUSE_EXISTING=1 e o arquivo $1 (o
# artefato que o comando produziria) ja exista e nao esteja vazio - nesse
# caso, so registra o skip no log e retorna sem executar nada.
run_step_reusable() {
    local artifact="$1" desc="$2"
    shift 2
    if [ "$REUSE_EXISTING" = "1" ] && [ -s "$artifact" ]; then
        echo ">>> ${desc}" | tee -a "$LOG_FILE"
        echo "    SKIP (${desc}) - artefato ja existe: $artifact" | tee -a "$LOG_FILE"
        return
    fi
    run_step "$desc" "$@"
}

echo "Regioes: ${REGIONS[*]}" | tee -a "$LOG_FILE"
echo "Log completo: $LOG_FILE"
echo ""

for region in "${REGIONS[@]}"; do
    echo "=====================================================" | tee -a "$LOG_FILE"
    echo "REGIAO: $region" | tee -a "$LOG_FILE"
    echo "=====================================================" | tee -a "$LOG_FILE"

    run_step_reusable "outputs/$region/spi_cache/spi_scale_3.pkl" "precompute-spi region=$region" \
        python3 main.py precompute-spi --region "$region"

    run_step_reusable "outputs/$region/autoencoder/model.pth" "train-ae region=$region" \
        python3 main.py train-ae --region "$region"

    run_step "grid-search (transfer learning) region=$region" \
        python3 main.py grid-search --region "$region" --use-transfer-learning --optimization-metric wi \
            --severity-weight "$SEVERITY_WEIGHT"

    run_step "grid-search (scratch) region=$region" \
        python3 main.py grid-search --region "$region" --scratch --optimization-metric wi \
            --severity-weight "$SEVERITY_WEIGHT"
done

echo ""
echo "Concluido. Log completo: $LOG_FILE"
if [ -s "$FAILED_LOG" ]; then
    echo "Houve falhas, veja: $FAILED_LOG"
    cat "$FAILED_LOG"
else
    echo "Nenhuma falha registrada."
fi
