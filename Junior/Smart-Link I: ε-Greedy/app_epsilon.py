import random

import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()

# Number of recommendations for order_id
recom = {}
# Link of a click_id with an offer_id
click_offer = {}
# Link of reward with an offer_id
reward_offer = {}
# Link of conversions with an offer_id
conversions_action = {}
offer_clicks = 0


@app.on_event("startup")
def startup_event():
    """
    A function that run before the application starts and clear dicts
    """
    recom.clear()
    click_offer.clear()
    reward_offer.clear()
    conversions_action.clear()
    offer_clicks = 0


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """
    Get feedback for particular click
    """
    # Response body consists of click ID
    # and accepted click status (True/False)
    offer_id = click_offer[click_id]
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": reward > 0.0,
        "reward": reward
    }
    reward_offer.setdefault(offer_id, 0)
    reward_offer[offer_id] += reward
    conversions_action.setdefault(offer_id, 0)
    conversions_action[offer_id] += response['is_conversion']
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """
    Return offer's statistics
    """
    try:
        cr = conversions_action.get(offer_id) / recom.get(offer_id)
    except:
        cr = 0
    try:
        rpc = reward_offer.get(offer_id) / recom.get(offer_id)
    except:
        rpc = 0

    response = {
        "offer_id": offer_id,
        "clicks": recom.get(offer_id, 0),
        "conversions": conversions_action.get(offer_id, 0),
        "reward": reward_offer.get(offer_id, 0),
        "cr": cr,
        "rpc": rpc,
    }
    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """
    Epsilon-greedy algorithm with epsilon=0.1
    :param click_id: int
    :param offer_ids: str
    :return: response: dict with click_id, offer_id, sampler
    """
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]
    global offer_clicks
    # Sample top offer ID
    p = np.random.random()
    epsilon = 0.1
    if p < epsilon:
        sampler = 'random'
    else:
        sampler = 'greedy'
    offer_clicks += 1
    if sampler == 'random':
        offer_id = random.choice(offers_ids)
    else:
        max_rpc = 0
        offer_id = 0
        for offer in offers_ids:
            try:
                rpc = reward_offer.get(offer) / recom.get(offer)
            except:
                rpc = 0
            if rpc > max_rpc:
                offer_id = offer
                max_rpc = rpc
        if max_rpc == 0:
            offer_id = offers_ids[0]

    click_offer[click_id] = offer_id
    recom.setdefault(offer_id, 0)
    recom[offer_id] += 1
    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": sampler,
    }
    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", port=8001)


if __name__ == "__main__":
    main()