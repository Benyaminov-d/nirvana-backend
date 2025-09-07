# Compass Score Configuration Guide

This document describes how to configure the Compass Score system through environment variables and database settings.

## 🎯 Overview

The Compass Score system now uses **centralized configuration** with hierarchical precedence:
1. **Database settings** (highest priority)
2. **Environment variables** (medium priority)  
3. **Sensible defaults** (fallback)

## 🔧 Environment Variables

### Core Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPASS_LAMBDA` | `2.25` | Loss-aversion parameter (behavioral finance) |
| `COMPASS_MU_LOW` | `0.02` | Lower anchor (2% - conservative bonds) |
| `COMPASS_MU_HIGH` | `0.18` | Upper anchor (18% - excellent stocks) |
| `COMPASS_DEFAULT_LOSS_TOLERANCE` | `0.25` | Default loss tolerance (25%) |

### Thresholds and Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPASS_MIN_SCORE_THRESHOLD` | `3000` | Minimum score for recommendations |
| `COMPASS_MAX_RESULTS` | `10` | Maximum number of results |
| `COMPASS_MIN_SAMPLE_SIZE` | `10` | Minimum data points for calibration |
| `COMPASS_DEFAULT_MEDIAN_MU` | `0.08` | Default median return (8%) |

### Calibration Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPASS_WINSOR_P_LOW` | `0.01` | Lower winsorization percentile (1%) |
| `COMPASS_WINSOR_P_HIGH` | `0.99` | Upper winsorization percentile (99%) |
| `COMPASS_ANCHOR_HD_LOW` | `0.05` | Lower Harrell-Davis percentile (5%) |
| `COMPASS_ANCHOR_HD_HIGH` | `0.95` | Upper Harrell-Davis percentile (95%) |
| `COMPASS_ANCHOR_MIN_SPREAD` | `0.02` | Minimum anchor spread (2%) |
| `COMPASS_ANCHOR_MAX_SPREAD` | `0.60` | Maximum anchor spread (60%) |

## 🔄 Configuration Hierarchy

### 1. Database Configuration (Highest Priority)
```sql
-- Check current anchors
SELECT category, version, mu_low, mu_high FROM compass_anchor 
ORDER BY created_at DESC LIMIT 5;

-- Create absolute anchors
INSERT INTO compass_anchor (category, version, mu_low, mu_high, ...) 
VALUES ('COMPASS_ABSOLUTE', '2025Q1', 0.02, 0.18, ...);
```

### 2. Environment Configuration
```bash
# Development
export COMPASS_MU_LOW=0.015
export COMPASS_MU_HIGH=0.20
export COMPASS_LAMBDA=2.5

# Production
COMPASS_MU_LOW=0.02
COMPASS_MU_HIGH=0.18
COMPASS_MIN_SCORE_THRESHOLD=4000
```

### 3. Docker Configuration
```yaml
# docker-compose.yml
environment:
  COMPASS_MU_LOW: ${COMPASS_MU_LOW:-0.02}
  COMPASS_MU_HIGH: ${COMPASS_MU_HIGH:-0.18}
  COMPASS_LAMBDA: ${COMPASS_LAMBDA:-2.25}
```

## 🛠️ Management Tools

### Creating Database Anchors
```bash
# Use the utility script
cd backend
python -m utils.create_compass_anchors

# With custom values
COMPASS_MU_LOW=0.015 COMPASS_MU_HIGH=0.22 python -m utils.create_compass_anchors
```

### API Management
```bash
# List current anchors
curl http://localhost:8000/compass/anchors

# Create anchors via API
curl -X POST http://localhost:8000/compass/anchors/calibrate \
  -H "Content-Type: application/json" \
  -d '{"category": "CUSTOM", "mu_values": [0.05, 0.08, 0.12, 0.15]}'
```

## 📊 Score Distribution Impact

### Conservative Configuration (Lower Scores)
```bash
COMPASS_MU_LOW=0.01    # 1% (very conservative)
COMPASS_MU_HIGH=0.25   # 25% (very aggressive)
# Result: Wider range, lower average scores
```

### Aggressive Configuration (Higher Scores)  
```bash
COMPASS_MU_LOW=0.03    # 3% (higher baseline)
COMPASS_MU_HIGH=0.15   # 15% (lower ceiling)  
# Result: Narrower range, higher average scores
```

## 🎯 Recommended Settings

### Demo Environment
```bash
COMPASS_MIN_SCORE_THRESHOLD=2500  # Show more products
COMPASS_MAX_RESULTS=15           # More variety
COMPASS_MU_HIGH=0.20             # Include growth stocks
```

### Production Environment
```bash
COMPASS_MIN_SCORE_THRESHOLD=4000  # Higher quality filter
COMPASS_MAX_RESULTS=10           # Focused results
COMPASS_MU_HIGH=0.18             # Conservative ceiling
```

### Risk-Averse Setup
```bash
COMPASS_LAMBDA=3.0               # Higher loss aversion
COMPASS_MU_HIGH=0.15             # Conservative ceiling
COMPASS_DEFAULT_LOSS_TOLERANCE=0.20  # Lower risk tolerance
```

## 🚨 Validation and Monitoring

### Health Checks
```bash
# Test configuration
curl http://localhost:8000/compass/selftest

# Verify scores
curl "http://localhost:8000/compass/score?mu=0.10&L=0.15&mu_low=0.02&mu_high=0.18&LT=0.25"
```

### Log Monitoring
```bash
# Watch for configuration logs
docker logs nirvana_backend 2>&1 | grep "anchor"
docker logs nirvana_backend 2>&1 | grep "Using.*anchors"
```

## ⚠️ Important Notes

1. **Restart Required**: Configuration changes require container restart
2. **Database Priority**: DB anchors override environment variables
3. **Validation**: Invalid values fall back to defaults with warnings
4. **Backward Compatibility**: Old hardcoded values completely eliminated
5. **Score Impact**: Anchor changes significantly affect score distribution

## 🔍 Troubleshooting

### No Results Returned
```bash
# Check threshold is not too high
echo $COMPASS_MIN_SCORE_THRESHOLD  # Should be ≤ 4000 for demo

# Check anchors are reasonable
curl http://localhost:8000/compass/anchors | jq '.[] | {mu_low, mu_high}'
```

### Scores Too Low/High
```bash
# Adjust anchor spread
export COMPASS_MU_LOW=0.01   # Lower baseline = higher scores
export COMPASS_MU_HIGH=0.25  # Higher ceiling = lower scores for mediocre products
```

### Configuration Not Applied
```bash
# Verify environment variables
docker exec nirvana_backend env | grep COMPASS

# Check configuration loading
docker logs nirvana_backend 2>&1 | grep -i "compass.*config"
```
