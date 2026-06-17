"""Tiny labeled corpus used for the demo and the evaluation harness.

Replace with the client's category schema + labeled examples. The pipeline
does not care about the specific labels - it learns centroids from whatever
(text, label) pairs you pass in.
"""

# (text, label) training examples - 4 categories, short inputs
TRAIN = [
    ("my order still has not arrived and tracking is stuck", "shipping"),
    ("where is my package, it was due yesterday", "shipping"),
    ("can you update the delivery address before it ships", "shipping"),
    ("the parcel shows delivered but i never received it", "shipping"),
    ("i was charged twice for the same item", "billing"),
    ("please refund the duplicate payment on my card", "billing"),
    ("my invoice total looks wrong this month", "billing"),
    ("how do i update the credit card on file", "billing"),
    ("the app crashes every time i open the settings page", "technical"),
    ("login button does nothing on the latest version", "technical"),
    ("getting a 500 error when i submit the form", "technical"),
    ("the export to pdf feature is broken", "technical"),
    ("do you offer a student discount", "sales"),
    ("what is included in the enterprise plan", "sales"),
    ("can i get a demo before i buy", "sales"),
    ("is there an annual subscription option", "sales"),
]

# held-out test set with gold labels (never seen during fit)
TEST = [
    ("my delivery is three days late and no update", "shipping"),
    ("the tracking number you gave me is invalid", "shipping"),
    ("i see an extra fee i did not authorize", "billing"),
    ("cancel my subscription and refund this month", "billing"),
    ("the dashboard wont load after the update", "technical"),
    ("i keep getting logged out every few minutes", "technical"),
    ("can you walk me through the pricing tiers", "sales"),
    ("do you have a free trial for teams", "sales"),
]
