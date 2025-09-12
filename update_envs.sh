#!/usr/bin/env bash
set -euo pipefail

RG=rg-nirvana
APP=func-nirvana
ENV_FILE="$HOME/Development/nirvana-cvar/nirvana_v1.0/nirvana-backend/envs"

val(){ grep -E "^$1=" "$ENV_FILE" | sed -E "s/^$1=//" || true; }

# Собираем список ключей, которые нужно прокинуть как ENV
KEYS="$(cat <<'EOF'
NVAR_SIMS
NVAR_SEED_NIG
NVAR_SEED_GHST
NVAR_GHST_DF
NVAR_GHST_FAMILY
NVAR_GHST_AUTO_SKEW
NVAR_EVAR_METHOD
NVAR_PRICE_FIELD
NVAR_ALLOW_CLOSE_FALLBACK
NVAR_RET_DECIMALS
NVAR_FEED_MAX_CVAR
NVAR_EQ_LOOKBACK_DAYS
NVAR_YEARS
NVAR_YEARS_DAYS
NVAR_MIN_YEARS
NVAR_ENFORCE_MIN_YEARS
NVAR_PRICE_LOG
NVAR_ENABLE_WEEKLY_RESAMPLE
NVAR_WINSOR_PLOW
NVAR_WINSOR_PHIGH
# Compass
COMPASS_LAMBDA
COMPASS_MU_LOW
COMPASS_MU_HIGH
COMPASS_MIN_SCORE_THRESHOLD
COMPASS_MAX_RESULTS
COMPASS_DEFAULT_LOSS_TOLERANCE
COMPASS_MIN_SAMPLE_SIZE
COMPASS_DEFAULT_MEDIAN_MU
EOF
)"

# Формируем массив KEY=VALUE (только для существующих в файле)
env_kv=()
while IFS= read -r K; do
  [[ -z "$K" || "$K" =~ ^# ]] && continue
  # значение без инлайн-комментариев, обрезать пробелы и снять внешние кавычки
  V="$(val "$K" | sed 's/[[:space:]]*#.*$//' | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | sed -E 's/^\"|\"$//g')"
  [[ -n "$V" ]] || continue
  env_kv+=("$K=$V")
done <<< "$KEYS"

# Применяем
az containerapp update -g "$RG" -n "$APP" --set-env-vars "${env_kv[@]}"

# Проверка
az containerapp show -g "$RG" -n "$APP" --query "properties.template.containers[0].env[?starts_with(name,'NVAR_') || starts_with(name,'COMPASS_')]" -o json