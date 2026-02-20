protoecg-pipeline extract  -n 10000
protoecg-pipeline tokenize -n 10000
protoecg-pipeline train    -n 10000
protoecg-pipeline evaluate -n 10000

- check fusion class ecg tokenization
- check normalization of ecg similarity (maybe not an issue with deciles)
- vitals
