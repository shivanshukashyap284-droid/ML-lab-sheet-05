# Q34. Analyze impact of hyperparameters on learning performance.
settings=[
    {"alpha":0.1,"gamma":0.99,"epsilon":0.1},
    {"alpha":0.5,"gamma":0.95,"epsilon":0.1},
    {"alpha":0.9,"gamma":0.90,"epsilon":0.2},
]
for s in settings:
    print(s)
print("\nAlpha controls update size; Gamma controls importance of future rewards; "
      "Epsilon controls exploration. Compare settings using the same number of episodes and seeds.")
