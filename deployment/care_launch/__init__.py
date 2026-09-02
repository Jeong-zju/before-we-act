"""Launch hardening shared by the MARS, DuoBench, and BiCoord CARE pipelines.

These modules exist so an unattended run on a rented host fails for real
reasons only: transient Hub faults are retried, and everything that can be
known before the first optimizer step is checked up front.
"""
