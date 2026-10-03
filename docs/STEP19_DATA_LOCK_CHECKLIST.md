# Step 19 Data-Lock Checklist

A real-data training lock is permitted only when every required gate below is satisfied.

## Source identity

- [ ] GEO/BioProject accession verified
- [ ] study publication/source verified
- [ ] processed input URLs explicitly declared
- [ ] every downloaded file SHA256 recorded
- [ ] acquisition timestamp recorded

## Biological hierarchy

- [ ] study family assigned
- [ ] study assigned
- [ ] subject/animal/donor verified
- [ ] sample/library verified
- [ ] cell/nucleus IDs preserved
- [ ] sample→subject mapping is one-to-one or explicitly documented
- [ ] subject→study mapping is unambiguous
- [ ] technical replicates identified
- [ ] repeated biological material identified

## Labels

- [ ] species verified
- [ ] assay verified
- [ ] tissue/region verified
- [ ] cell type verified
- [ ] maturation verified or explicitly not applicable
- [ ] injury verified or explicitly not applicable
- [ ] condition verified
- [ ] timepoint verified
- [ ] unresolved labels excluded

## Expression integrity

- [ ] matrix dimensions match metadata
- [ ] observation IDs are unique
- [ ] gene IDs are unique after reconciliation
- [ ] matrix contains no non-finite values
- [ ] no silent densification during ingestion
- [ ] explicit gene-ID mapping recorded
- [ ] cross-species orthology mapping frozen before merge
- [ ] modality boundaries preserved

## QC

- [ ] cell/nucleus library-size QC
- [ ] detected-gene QC
- [ ] mitochondrial/ribosomal QC where biologically appropriate
- [ ] doublet/ambient-RNA policy documented where applicable
- [ ] minimum sample cell-count gate applied
- [ ] sample-level outlier review completed
- [ ] QC thresholds fixed before locked evaluation

## Leakage control

- [ ] no study crosses partitions
- [ ] no study family crosses partitions
- [ ] no subject crosses partitions
- [ ] no sample crosses partitions
- [ ] preprocessing fit on training data only
- [ ] feature selection fit on training data only
- [ ] categorical mappings fit on training data only
- [ ] validation/test remain untouched until evaluation

## Lock

- [ ] reviewed split manifest approved by human reviewer
- [ ] split fingerprint generated
- [ ] expression fingerprint generated
- [ ] metadata fingerprint generated
- [ ] lock manifest generated
- [ ] candidate registry changed from candidate only after review

**Important:** CI success demonstrates software correctness, not biological validity. The lock is a scientific data decision and requires review.
