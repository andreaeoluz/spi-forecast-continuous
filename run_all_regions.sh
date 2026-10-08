#!/usr/bin/env bash
#
# Roda inference + analyze para as 5 regioes do Brasil em TODAS as
# combinacoes da grade: p x q x model-type (spi-scale fixo)
# (5 regioes x 4 p x 5 q x 2 tipos = 200 inferencias + 200 analises).
#
# Horizontes longos tem poucas janelas no teste (2023-2024, 24 meses): o
# numero de meses-alvo avaliados e 24 - (p + q) + 1, por exemplo 21 para
# p=3,q=1 e apenas 1 para p=12,q=12. Interprete as metricas desses casos
# com cautela.
#
# Depois de rodar, `python3 extract_test_period.py` consolida as metricas em
# test_period_full.csv (todas as combinacoes) e test_period.csv (q=1).
#
# Uso:
#   ./run_all_regions.sh
#
# Rode a partir da raiz do pacote (onde ficam main.py e a pasta inference/),
# ex.: drought_forecast_regression/
#
# Se algum comando falhar, o script registra o erro em failed_runs.log e
# continua com as demais combinacoes (nao usa 'set -e').

set -uo pipefail

REGIONS=("Sul" "Sudeste" "Nordeste" "Centro-Oeste" "Norte")
PS=(3 6 9 12)
QS=(1 3 6 9 12)
MODEL_TYPES=("pretrained" "scratch")

LOG_FILE="run_all_regions_$(date +%Y%m%d_%H%M%S).log"
FAILED_LOG="failed_runs.log"
: > "$FAILED_LOG"

TOTAL=$(( ${#REGIONS[@]} * ${#MODEL_TYPES[@]} * ${#PS[@]} * ${#QS[@]} ))
echo "Combinacoes por etapa: $TOTAL (inferencia + analise)" | tee -a "$LOG_FILE"

run_step() {
    local desc="$1"
    shift
    echo ">>> ${desc}" | tee -a "$LOG_FILE"
    if ! "$@" >>"$LOG_FILE" 2>&1; then
        echo "FALHOU: $desc" | tee -a "$FAILED_LOG"
    fi
}

# ---------------------------------------------------------------------------
# 1) Inferencia
# ---------------------------------------------------------------------------
for region in "${REGIONS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        for p in "${PS[@]}"; do
            for q in "${QS[@]}"; do
                run_step "inference region=$region p=$p q=$q model_type=$model_type" \
                    python3 main.py inference \
                        --region "$region" \
                        --p "$p" --q "$q" \
                        --model-type "$model_type"
            done
        done
    done
done

# ---------------------------------------------------------------------------
# 2) Analise
# ---------------------------------------------------------------------------
for region in "${REGIONS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        for p in "${PS[@]}"; do
            for q in "${QS[@]}"; do
                run_step "analyze region=$region p=$p q=$q model_type=$model_type" \
                    python3 -m inference.analyze \
                        --region "$region" \
                        --p "$p" --q "$q" \
                        --model-type "$model_type"
            done
        done
    done
done

echo ""
echo "Concluido. Log completo: $LOG_FILE"
if [ -s "$FAILED_LOG" ]; then
    echo "Houve falhas, veja: $FAILED_LOG"
    cat "$FAILED_LOG"
else
    echo "Nenhuma falha registrada."
fi
