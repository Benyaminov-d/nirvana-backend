# Nirvana System Maintenance API

## Overview

The Maintenance API provides endpoints for system administration, monitoring, and diagnostics. This API allows operators to:

- Reset circuit breakers when they're in OPEN state
- Verify and validate Compass anchors
- Check data integrity
- Get comprehensive system status

These APIs are secured with the same authentication as other endpoints, requiring `pub` or `basic` authentication.

## API Endpoints

### 1. Reset Circuit Breaker

```
POST /api/maintenance/reset-circuit-breaker
```

Resets circuit breakers for external services, allowing API calls to resume after failures.

**Query Parameters:**

- `service` (string, optional): Specific service to reset (e.g., `eodhd_historical_prices`)
- `all_services` (boolean, optional): If true, reset all circuit breakers

**Response:**

```json
{
  "success": true,
  "reset_count": 1,
  "reset_services": {
    "eodhd_historical_prices": "reset"
  }
}
```

**Example Usage:**

```bash
# Reset a specific service
curl -X POST "https://your-domain.com/api/maintenance/reset-circuit-breaker?service=eodhd_historical_prices" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Reset all circuit breakers
curl -X POST "https://your-domain.com/api/maintenance/reset-circuit-breaker?all_services=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 2. Verify Anchors

```
GET /api/maintenance/verify-anchors
```

Verifies and validates the calibration and quality of Compass anchors.

**Query Parameters:**

- `calibrate_if_invalid` (boolean, optional): If true, automatically recalibrates invalid anchors

**Response:**

```json
{
  "success": true,
  "critical_issue": false,
  "issues": {},
  "missing_critical": [],
  "anchors_count": 12,
  "current_quarter": "2025Q3",
  "anchors": {
    "GLOBAL:US": {
      "mu_low": 0.02,
      "mu_high": 0.18,
      "median_mu": 0.06,
      "version": "2025Q3",
      "current": true,
      "issues": []
    },
    "HARVARD-US": {
      "mu_low": 0.03,
      "mu_high": 0.17,
      "median_mu": 0.07,
      "version": "2025Q3",
      "current": true,
      "issues": []
    }
    // Additional anchors...
  }
}
```

**Example Usage:**

```bash
# Verify anchors without recalibration
curl "https://your-domain.com/api/maintenance/verify-anchors" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Verify anchors and recalibrate if needed
curl "https://your-domain.com/api/maintenance/verify-anchors?calibrate_if_invalid=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. Data Integrity Check

```
GET /api/maintenance/data-integrity
```

Verifies the integrity of critical data in the system.

**Response:**

```json
{
  "success": true,
  "issues": [],
  "warnings": [
    "Few recent CVaR snapshots: 95 in last 7 days"
  ],
  "cvar_data_count": 12500
}
```

**Example Usage:**

```bash
curl "https://your-domain.com/api/maintenance/data-integrity" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. System Status

```
GET /api/maintenance/system-status
```

Gets a comprehensive overview of system status, including circuit breakers, anchors, and data integrity.

**Response:**

```json
{
  "timestamp": 1694521234.587,
  "components": {
    "database": {
      "connected": true,
      "error": null
    },
    "circuit_breakers": {
      "eodhd_historical_prices": {
        "state": "closed",
        "failures": 0,
        "last_failure": 1694520000.123,
        "last_success": 1694521200.456,
        "open_duration": 0
      }
    },
    "anchors": {
      "success": true,
      "critical_issue": false,
      "issues": {},
      "anchors_count": 12,
      "current_quarter": "2025Q3"
    },
    "data_integrity": {
      "success": true,
      "issues": [],
      "warnings": [
        "Few recent CVaR snapshots: 95 in last 7 days"
      ],
      "cvar_data_count": 12500
    }
  },
  "critical_issues": false
}
```

**Example Usage:**

```bash
curl "https://your-domain.com/api/maintenance/system-status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Operational Best Practices

### Circuit Breakers

1. **Monitoring**: Regularly check circuit breaker status via `/api/maintenance/system-status`
2. **Reset Timing**: Reset circuit breakers only after addressing the underlying issue
3. **Logging**: After resetting a circuit breaker, monitor logs for new failures

### Anchor Management

1. **Regular Validation**: Verify anchors at the beginning of each quarter
2. **Calibration Timing**: Recalibrate anchors when:
   - A new quarter begins
   - A significant change occurs in market conditions
   - New data has been added for a significant number of instruments
3. **Quality Issues**: Pay attention to issues like:
   - Small spreads (< 0.05)
   - Large spreads (> 0.40)
   - Extreme values (mu_low < -0.10, mu_high > 0.50)

### Data Integrity

1. **Warning Resolution**: Actively address warnings before they become critical issues
2. **Data Coverage**: Maintain adequate CVaR data coverage across all markets
3. **Data Freshness**: Regularly update market data to avoid stale snapshots

## Environment Variables

The following environment variables control behavior of maintenance features:

- `STARTUP_RESET_CIRCUIT_BREAKERS`: Set to "1" to reset circuit breakers on startup (default: "1")
- `STARTUP_VALIDATE_DATA`: Set to "1" to validate data integrity on startup (default: "1")
- `COMPASS_ANCHOR_SCOPE`: Controls anchor calibration scope ("auto", "validated", "global", "full")
- `STARTUP_COMPASS_ANCHORS`: Set to "1" to calibrate anchors on startup (default: "1")

## Troubleshooting

### Circuit Breaker Issues

1. **Open State Persists**: If a circuit breaker remains in OPEN state after reset:
   - Check API credentials for the external service
   - Verify the external service is operational
   - Look for connection or network issues

2. **Frequent Failures**: If circuit breakers frequently switch to OPEN:
   - Increase failure threshold or recovery timeout
   - Check for API rate limiting issues
   - Consider implementing backoff strategies

### Anchor Issues

1. **Calibration Failures**: If anchor calibration fails:
   - Verify sufficient data availability
   - Check for data quality issues
   - Examine database connectivity

2. **Suspicious Values**: If anchor values are suspicious:
   - Review input data for outliers
   - Check calibration parameters
   - Consider manual calibration with adjusted parameters

### Data Integrity Issues

1. **Missing Data**: If critical data is missing:
   - Verify external data sources
   - Check import processes
   - Examine database indexing and queries

2. **Stale Data**: If data is stale:
   - Check data update schedules
   - Verify synchronization processes
   - Examine ETL pipelines
