
# Naive Bayes - Updating probabilities from a list of actions

The idea of this notebook is to demonstrate how a list of actions can be used the prior probability of something being true or not.

As an example scenario, Let's say that we are monitoring a web application, and we want to identify if a user is a buyer (1) or something else (0).

To keep this simple, I will imit the possible actions to 4 actions: search, sell, buy, browse.



```python
# Defining the tests and their probabilities

likelyhood = {}

likelyhood['search'] = { # likelihood that a buyer will use the search function
    1 : {
        1: 0.6, # P(buyer | search) (True Positive)
        0: 0.3 # P(-buyer | search) (False Positive)
    },
    0: {
        1: 0.4, # P(buyer | -search) (True Negative)
        0: 0.7 # P(-buyer | -search) (False Negative)
    }
}

likelyhood['sell'] = { # likelihoods that a buyer will sell an item
    1 : {
        1: 0.05, # P(buyer | sell) (True Positive)
        0: 0.5 # P(-buyer | sell) (False Positive)
    },
    0: {
        1: 0.95, # P(buyer | -sell) (True Negative)
        0: 0.5 # P(-buyer | -sell) (False Negative)
    }
}

likelyhood['buy'] = { # likelihoods that a buyer will buy an item
    1 : {
        1: 0.3, # P(buyer | buy) (True Positive)
        0: 0.2 # P(-buyer | buy) (False Positive)
    },
    0: {
        1: 0.7, # P(buyer | -buy) (True Negative)
        0: 0.8 # P(-buyer | -buy) (False Negative)
    }
}

likelyhood['view'] = { # likelihoods that a buyer will view the details of an item
    1 : {
        1: 0.9, # P(buyer | view) (True Positive)
        0: 0.85 # P(-buyer | view) (False Positive)
    },
    0: {
        1: 0.1, # P(buyer | -view) (True Negative)
        0: 0.15 # P(-buyer | -view) (False Negative)
    }
}
```


```python
def update_probability(prior_probability, test_name, distribution, test_result):
    # Account for evidence
    
    likelihood = distribution[test_name][test_result][1]
    non_likelihood = distribution[test_name][test_result][0]
    
    if test_result:
        test_status = 'positive'
    else:
        test_status = 'negative'
        
    numerator = likelihood * prior_probability
    denominator = (likelihood * prior_probability) + (non_likelihood * (1 - prior_probability))
    
    conditional_probability = numerator / denominator
    
    print('\t* {} is {}. Prior probability is: {:.2f}%. Updated probability is: {:.2f}%'.format(test_name, test_status, 100 * prior_probability, 100 * conditional_probability))
    
    return conditional_probability
```


```python
def analyse_events(prior, events, legals):
    print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))
    posterior = prior
    for ee in events:
        posterior = update_probability(posterior, ee, likelyhood, 1)
    for ll in legal_actions:
        if ll not in events:
            posterior = update_probability(posterior, ll, likelyhood, 0)
    return posterior
```


```python
# List of legal actions
legal_actions = likelyhood.keys()

# Prior: What is our initial belief that the user is a buyer
prior = 0.5 
```

# Updating our bliefs, given evidences

Let's say that we are observing a chain of events. For each of these events, we want to update our belief about the user.


```python
events = ['search', 'view', 'search', 'view', 'view', 'buy']

posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))
```

    Given the evidences 'search,view,search,view,view,buy', what is the posterior probability that our user is a buyer?
    	* search is positive. Prior probability is: 50.00%. Updated probability is: 66.67%
    	* view is positive. Prior probability is: 66.67%. Updated probability is: 67.92%
    	* search is positive. Prior probability is: 67.92%. Updated probability is: 80.90%
    	* view is positive. Prior probability is: 80.90%. Updated probability is: 81.77%
    	* view is positive. Prior probability is: 81.77%. Updated probability is: 82.60%
    	* buy is positive. Prior probability is: 82.60%. Updated probability is: 87.69%
    	* sell is negative. Prior probability is: 87.69%. Updated probability is: 93.12%
    Probability that our user is a buyer is: 93.12%



```python
events = ['search', 'view', 'view', 'view', 'sell', 'sell']

posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))

```

    Given the evidences 'search,view,view,view,sell,sell', what is the posterior probability that our user is a buyer?
    	* search is positive. Prior probability is: 50.00%. Updated probability is: 66.67%
    	* view is positive. Prior probability is: 66.67%. Updated probability is: 67.92%
    	* view is positive. Prior probability is: 67.92%. Updated probability is: 69.16%
    	* view is positive. Prior probability is: 69.16%. Updated probability is: 70.36%
    	* sell is positive. Prior probability is: 70.36%. Updated probability is: 19.19%
    	* sell is positive. Prior probability is: 19.19%. Updated probability is: 2.32%
    	* buy is negative. Prior probability is: 2.32%. Updated probability is: 2.04%
    Probability that our user is a buyer is: 2.04%



```python
events = ['buy', 'sell', 'view', 'sell', 'buy', 'view']

posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))
```

    Given the evidences 'buy,sell,view,sell,buy,view', what is the posterior probability that our user is a buyer?
    	* buy is positive. Prior probability is: 50.00%. Updated probability is: 60.00%
    	* sell is positive. Prior probability is: 60.00%. Updated probability is: 13.04%
    	* view is positive. Prior probability is: 13.04%. Updated probability is: 13.71%
    	* sell is positive. Prior probability is: 13.71%. Updated probability is: 1.56%
    	* buy is positive. Prior probability is: 1.56%. Updated probability is: 2.33%
    	* view is positive. Prior probability is: 2.33%. Updated probability is: 2.46%
    	* search is negative. Prior probability is: 2.46%. Updated probability is: 1.42%
    Probability that our user is a buyer is: 1.42%

