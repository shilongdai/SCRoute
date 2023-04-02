import "bootstrap/dist/css/bootstrap.css";
import "./index.css"
import React, {useState} from "react";
import {Spinner} from 'react-bootstrap';
import Header from "./Header";
import InputSection from "./InputSection";
import ResultSection from "./ResultSection";


const opt_endpoint = "/optimize"


function buildRequest(range, steps, cargo, restrictions, commodities, locations) {
    const req = {
        max_range: range,
        max_cargo: cargo * 100,
        stops: steps,
        max_commodity: {},
        restrictions: restrictions,
        blk_locations: locations
    }
    for (const c of commodities) {
        req.max_commodity[c.name] = c.amount * 0.01
    }
    return req
}


function App() {
    const [highLevelPlan, setHighLevelPlan] = useState(null);
    const [tradeRoute, setTradeRoute] = useState(null);
    const [restrictions, setRestrictions] = useState({})
    const [loading, setLoading] = useState(false);

    function setResult(plan, route) {
        setHighLevelPlan(plan);
        setTradeRoute(route);
        setLoading(false)
    }

    const handleSubmit = (range, steps, cargo, commodities, locations) => {
        const requestOptions = {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(buildRequest(range, steps, cargo, restrictions, commodities, locations))
        };
        setLoading(true)
        fetch(opt_endpoint, requestOptions).then(response => response.json()).then(data => setResult(data.plan, data.routes))

    };

    const addBlacklist = (newRestrictions, transactions) => {
        if(transactions) {
            for (const t of transactions) {
                if (!Object.hasOwn(newRestrictions, t.commodity)) {
                    newRestrictions[t.commodity] = {}
                }
                newRestrictions[t.commodity][t.location] = 0
            }
        }
    }

    const handleBlacklist = () => {
        const newRestrictions = structuredClone(restrictions)
        if(highLevelPlan) {
            addBlacklist(newRestrictions, highLevelPlan.buyTransactions)
            addBlacklist(newRestrictions, highLevelPlan.sellTransactions)
        }
        setRestrictions(newRestrictions)
    }

    const handleReset = () => {
        setRestrictions({})
    }

    return (<div className="container">
            <Header/>
            <div className="row">
                <div className="col-lg-6">
                    <InputSection onSubmit={handleSubmit} onBlacklistReset={handleReset}
                                  onBlacklist={handleBlacklist} lockForm={loading}/>
                </div>
                <div className="col-lg-6 mt-4 mt-lg-2">
                    {loading && (
                        <div className="d-flex justify-content-center">
                            <p>Loading...</p>
                            <div className="my-4">
                                <div className="spinner-border text-primary" role="status">
                                    <span className="visually-hidden">Loading...</span>
                                </div>
                            </div>
                        </div>
                    )}
                    {highLevelPlan && tradeRoute && !loading && (
                        <ResultSection highLevelPlan={highLevelPlan} tradeRoute={tradeRoute}/>)}
                </div>
            </div>
        </div>

    );
}

export default App;