# Security policy

Apollon is research software for intrusion detection. If you find a vulnerability in the
code (for example, unsafe handling of input files), please report it privately through
[GitHub's private vulnerability reporting](https://github.com/antoniopaya22/apollon/security/advisories/new)
rather than in a public issue.

Model files saved with `save_model` use joblib (pickle). Only load model files from
sources you trust.
