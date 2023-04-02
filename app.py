from flask import Flask, request, jsonify, session, send_from_directory
from flask_session import Session
import uuid
from optimize import *


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


class BadRequestException(Exception):

    def __init__(self):
        pass


def convert_transactions(ts):
    return [{
        "location": t.loc_idx,
        "commodity": t.com_idx,
        "amount": t.amount
    } for t in ts]


def convert_plan(plan):
    plan_result = {"revenue": plan.revenue, "cost": plan.cost, "buyTransactions": convert_transactions(plan.buy),
                   "sellTransactions": convert_transactions(plan.sell)}
    return plan_result


def convert_route(routes):
    result = []
    for r in routes:
        temp = {
            "startLocation": r.start,
            "endLocation": r.end,
            "buyTransactions": convert_transactions(r.buy),
            "sellTransactions": convert_transactions(r.sell)
        }
        result.append(temp)
    return result


@app.errorhandler(BadRequestException)
def handle_bad_request(arg):
    return "Bad Request", 400


@app.route("/")
def serve_home_page():
    return send_from_directory("static", "index.html")


@app.route("/locations")
def retrieve_locations():
    locs = get_valid_shops()
    return jsonify(locs)


@app.route("/commodities")
def retrieve_commodities():
    coms = get_valid_coms()
    return jsonify(coms)


@app.route('/optimize', methods=["POST"])
def optimize():
    if "id" not in session:
        session["id"] = uuid.uuid4()
    current_id = session["id"]
    trade_info = request.json
    try:
        max_range = int(trade_info["max_range"])
        max_cargo = int(trade_info["max_cargo"])
        stops = int(trade_info["stops"])

        max_commodities = {}
        if "max_commodity" in trade_info:
            max_commodities = trade_info["max_commodity"]

        blk_locs = []
        if "blk_locations" in trade_info:
            blk_locs = trade_info["blk_locations"]

        restrictions = {}
        if "restrictions" in trade_info:
            restrictions = trade_info["restrictions"]
    except (KeyError, ValueError):
        raise BadRequestException()
    if max_range <= 0:
        raise BadRequestException()
    if "past_plans" not in session:
        session["past_plans"] = []

    plan, routes = get_solver(current_id)(max_cargo, stops, max_range, blk_locs, max_commodities, restrictions)
    final_map = {
        "plan": convert_plan(plan),
        "routes": convert_route(routes)
    }

    return jsonify(final_map)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
