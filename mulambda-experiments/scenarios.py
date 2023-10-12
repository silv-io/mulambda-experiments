DUMMY_MODELS = (
    {
        "Name": "FastBad",
        "D": 10,
        "alpha": 0.001,
        "beta": 0.1,
        "J": 5,
        "confidences": [0.5, 0.3, 0.1],
        "gamma": 0.1,
    },
    {
        "Name": "FastOK",
        "D": 15,
        "alpha": 0.002,
        "beta": 0.35,
        "J": 5,
        "confidences": [0.8, 0.5, 0.3],
        "gamma": 0.2,
    },
    {
        "Name": "FastGood",
        "D": 20,
        "alpha": 0.004,
        "beta": 0.7,
        "J": 5,
        "confidences": [0.9, 0.7, 0.5],
        "gamma": 0.3,
    },
    {
        "Name": "MidBad",
        "D": 50,
        "alpha": 0.01,
        "beta": 0.1,
        "J": 30,
        "confidences": [0.55, 0.35, 0.15],
        "gamma": 0.4,
    },
    {
        "Name": "MidOK",
        "D": 75,
        "alpha": 0.02,
        "beta": 0.3,
        "J": 30,
        "confidences": [0.85, 0.55, 0.35],
        "gamma": 0.5,
    },
    {
        "Name": "MidGood",
        "D": 100,
        "alpha": 0.04,
        "beta": 0.7,
        "J": 30,
        "confidences": [0.95, 0.75, 0.55],
        "gamma": 0.6,
    },
    {
        "Name": "SlowBad",
        "D": 150,
        "alpha": 0.06,
        "beta": 0.1,
        "J": 60,
        "confidences": [0.6, 0.4, 0.2],
        "gamma": 0.7,
    },
    {
        "Name": "SlowMid",
        "D": 175,
        "alpha": 0.07,
        "beta": 0.35,
        "J": 60,
        "confidences": [0.9, 0.6, 0.4],
        "gamma": 0.8,
    },
    {
        "Name": "SlowGood",
        "D": 200,
        "alpha": 0.08,
        "beta": 0.7,
        "confidences": [1.0, 0.8, 0.6],
        "J": 60,
        "gamma": 0.9,
    },
)

MODEL_IDS = range(len(DUMMY_MODELS))

SCENARIOS = {
    "parity": [
        [0, 4, 5, 8],
        [0, 4, 5, 8],
        [0, 4, 5, 8],
        [0, 4, 5, 8],
    ],
    "hyperlocal": [MODEL_IDS, [], [], []],
    "logical": [
        [0, 1, 5, 8],
        [3, 4, 5, 8],
        [5, 6, 7, 8],
        [0, 6, 7, 8],
    ],
    "arbitrary": [
        [2, 6, 1, 8],
        [4, 0, 7, 3],
        [6, 8, 1, 0],
        [3, 7, 6, 2],
    ],
}

NODES = range(4)

if __name__ == "__main__":
    import random

    test_model = DUMMY_MODELS[0]
    values = []
    for i in range(100):
        base_delay = test_model["D"] / 1000
        input_size = 10
        size_impact = test_model["alpha"]
        concurrency_impact = test_model["beta"]
        concurrency = 1
        jitter = random.uniform(0, test_model["J"] / 1000)
        values.append(
            base_delay
            + input_size * size_impact
            + concurrency * concurrency_impact
            + jitter
        )

    print(sum(values) / len(values))
