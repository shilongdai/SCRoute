import React from 'react';

const ResultSection = ({highLevelPlan, tradeRoute}) => {
    // Create a new array to group transactions by location
    const transactionsByLocation = [];
    for (const transaction of highLevelPlan.buyTransactions.concat(highLevelPlan.sellTransactions)) {
        const locationName = transaction.location;
        let location = transactionsByLocation.find(loc => loc.name === locationName);
        if (!location) {
            location = {name: locationName, buyTransactions: [], sellTransactions: []};
            transactionsByLocation.push(location);
        }
        if (highLevelPlan.buyTransactions.includes(transaction)) {
            location.buyTransactions.push(transaction);
        } else {
            location.sellTransactions.push(transaction);
        }
    }

    // Calculate the total cost and revenue of the high level plan
    const totalCost = highLevelPlan.cost
    const totalRevenue = highLevelPlan.revenue

    // Calculate the profit of the high level plan
    const profit = totalRevenue - totalCost;

    return (
        <div className="result-section container">
            <h2>Plan</h2>
            <div className="row mb-4">
                <div className="col-md-12">
                    <div className="d-flex flex-column align-items-center">
                        <div className="row w-100">
                            <div className="col-md-4 mb-3 mb-md-0">
                                <div className="card bg-light">
                                    <div className="card-body">
                                        <h5 className="card-title">Total Cost</h5>
                                        <p className={`card-text text-danger`}>
                                            {totalCost >= 0 ? `$${totalCost}` : `-$${Math.abs(totalCost)}`}
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div className="col-md-4 mb-3 mb-md-0">
                                <div className="card bg-light">
                                    <div className="card-body">
                                        <h5 className="card-title">Total Revenue</h5>
                                        <p className={`card-text text-success`}>
                                            {totalRevenue >= 0 ? `$${totalRevenue}` : `-$${Math.abs(totalRevenue)}`}
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div className="col-md-4">
                                <div className="card bg-light">
                                    <div className="card-body">
                                        <h5 className="card-title">Profit</h5>
                                        <p className={`card-text text-${profit >= 0 ? 'success' : 'danger'}`}>
                                            {profit >= 0 ? `$${profit}` : `-$${Math.abs(profit)}`}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {transactionsByLocation.map(location => (
                <div key={location.name} className="row mb-4">
                    <div className="col-md-12">
                        <div className="card">
                            <div className="card-header">
                                {location.name}
                            </div>
                            <div className="card-body">
                                <table className="table">
                                    <thead>
                                    <tr>
                                        <th>Commodity</th>
                                        <th>Amount</th>
                                        <th>Type</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {location.buyTransactions.map(transaction => (
                                        <tr key={transaction.commodity}>
                                            <td>{transaction.commodity}</td>
                                            <td>{(transaction.amount / 100).toFixed(1)}</td>
                                            <td>Buy</td>
                                        </tr>
                                    ))}
                                    {location.sellTransactions.map(transaction => (
                                        <tr key={transaction.commodity}>
                                            <td>{transaction.commodity}</td>
                                            <td>{(transaction.amount / 100).toFixed(1)}</td>
                                            <td>Sell</td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
            <div className="row mb-4">
                <h2>Route</h2>

                {tradeRoute.map(route => (
                    <div className="col-md-12 mb-4">
                        <div className="card">
                            <div className="card-header">
                                {route.endLocation}
                            </div>
                            <div className="card-body">
                                <table className="table">
                                    <thead>
                                    <tr>
                                        <th>Commodity</th>
                                        <th>Amount</th>
                                        <th>Type</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {route.buyTransactions.map(transaction => (
                                        <tr key={transaction.commodity}>
                                            <td>{transaction.commodity}</td>
                                            <td>{(transaction.amount / 100).toFixed(1)}</td>
                                            <td>Buy</td>
                                        </tr>
                                    ))}
                                    {route.sellTransactions.map(transaction => (
                                        <tr key={transaction.commodity}>
                                            <td>{transaction.commodity}</td>
                                            <td>{(transaction.amount / 100).toFixed(1)}</td>
                                            <td>Sell</td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                ))}

            </div>
        </div>
    );
};

export default ResultSection;

