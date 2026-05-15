"""AWS reference provider.

Concrete implementations of Ray (and, in Phase 6, Dask) cluster
lifecycle management on AWS. The orchestration layer imports these as
duck-typed context managers — there is no abstract ``Provider`` base
class.
"""
