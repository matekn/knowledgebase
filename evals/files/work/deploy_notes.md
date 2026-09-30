# Deploy notes (work)

Staging deploys happen automatically on merge to main.
Production deploys require a manual approval in the pipeline.
Rollback is `deploy --revert <service>`.
