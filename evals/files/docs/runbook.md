# Deployment Runbook

## Rolling back the payments service
If the payments deploy misbehaves, roll back by running `deploy --revert payments`.
The on-call engineer must page the database team before reverting schema migrations.

## Database backups
Nightly backups run at 02:00 UTC via the `pg_dump` cron. Backups are retained for 30 days
in the eu-west-1 bucket. To restore, use `restore --from <timestamp>`.

## Incident severity levels
- SEV1: full outage, customer-facing. Page immediately.
- SEV2: degraded performance. Respond within 30 minutes.
- SEV3: minor bug, no customer impact. Handle next business day.
