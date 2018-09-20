
# Naive Bayes - Updating probabilities from a list of actions

The idea of this notebook is to demonstrate how a list of actions can be used to update the probabilities of something being true or not.

## Scenario
As an example scenario, Let's say that we are monitoring a web application, and we want to identify if a user is a buyer (1) or something else (0).

## Observable Variables
To keep this simple, I will limit the possible actions to 4 actions: search, sell, buy, browse.



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
    
    # Account for evidences  
    if test_name in distribution.keys(): ## First, let's check if we have probabilities for the requested test_name
        ## If so, we use these probabilities instead of the default ones
        likelihood = distribution[test_name][test_result][1]
        non_likelihood = distribution[test_name][test_result][0]
        if test_result:
            test_status = 'positive'
        else:
            test_status = 'negative'
    else:  ## if not, we go for generic values
        likelihood = 0.5
        non_likelihood = 0.5
        test_status = 'unknown'
      
    numerator = likelihood * prior_probability
    denominator = (likelihood * prior_probability) + (non_likelihood * (1 - prior_probability))
    
    conditional_probability = numerator / denominator
    
    print('\t* "{}" is {}. Prior probability is: {:.2f}%. Updated probability is: {:.2f}%'.format(test_name, test_status, 100 * prior_probability, 100 * conditional_probability))
    
    return conditional_probability
```


```python
def analyse_events(prior, events, legals):
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

## Buyer profile


```python
events = ['search', 'view', 'search', 'view', 'view', 'buy']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))
posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))
```

    Given the evidences 'search,view,search,view,view,buy', what is the posterior probability that our user is a buyer?
    	* "search" is positive. Prior probability is: 50.00%. Updated probability is: 66.67%
    	* "view" is positive. Prior probability is: 66.67%. Updated probability is: 67.92%
    	* "search" is positive. Prior probability is: 67.92%. Updated probability is: 80.90%
    	* "view" is positive. Prior probability is: 80.90%. Updated probability is: 81.77%
    	* "view" is positive. Prior probability is: 81.77%. Updated probability is: 82.60%
    	* "buy" is positive. Prior probability is: 82.60%. Updated probability is: 87.69%
    	* "sell" is negative. Prior probability is: 87.69%. Updated probability is: 93.12%
    Probability that our user is a buyer is: 93.12%


## Seller profile

Notice that our test is trying to identify if the user is a buyer, not if s/he is a seller.
If we wanted to know the probability that the user is a seller, we would need different likelihood data.

Still, let's see how this model reacts.


```python
events = ['search', 'view', 'view', 'view', 'sell', 'sell']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))
posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))

```

    Given the evidences 'search,view,view,view,sell,sell', what is the posterior probability that our user is a buyer?
    	* "search" is positive. Prior probability is: 50.00%. Updated probability is: 66.67%
    	* "view" is positive. Prior probability is: 66.67%. Updated probability is: 67.92%
    	* "view" is positive. Prior probability is: 67.92%. Updated probability is: 69.16%
    	* "view" is positive. Prior probability is: 69.16%. Updated probability is: 70.36%
    	* "sell" is positive. Prior probability is: 70.36%. Updated probability is: 19.19%
    	* "sell" is positive. Prior probability is: 19.19%. Updated probability is: 2.32%
    	* "buy" is negative. Prior probability is: 2.32%. Updated probability is: 2.04%
    Probability that our user is a buyer is: 2.04%


## Power whatever profile

Let's say that our user is actually a hobbyist trying to buy cheap and resell at a higher price, and therefore make money over other sellers who are willing to sell for cheaper than they should. How would our model react?


```python
events = ['buy', 'sell', 'view', 'sell', 'buy', 'view']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))
posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))
```

    Given the evidences 'buy,sell,view,sell,buy,view', what is the posterior probability that our user is a buyer?
    	* "buy" is positive. Prior probability is: 50.00%. Updated probability is: 60.00%
    	* "sell" is positive. Prior probability is: 60.00%. Updated probability is: 13.04%
    	* "view" is positive. Prior probability is: 13.04%. Updated probability is: 13.71%
    	* "sell" is positive. Prior probability is: 13.71%. Updated probability is: 1.56%
    	* "buy" is positive. Prior probability is: 1.56%. Updated probability is: 2.33%
    	* "view" is positive. Prior probability is: 2.33%. Updated probability is: 2.46%
    	* "search" is negative. Prior probability is: 2.46%. Updated probability is: 1.42%
    Probability that our user is a buyer is: 1.42%


The probability of our user being a buyer is really low, even if he is buying a lot. This tells us that he is doing something else that doesn't fit the normal behaviour. This could worth investigating.

## What happen if we test with unknown variables?

Let's say that our user is trying to commit fraud, what would happen?

Because the variables are unknown, we can't update positively or negatively the posterior. We could of course configure a default value that would move the results in one direction or the other, but this could also generate false results.

Because of this, the update_probability function will use a 50%/50% probability, which has the impact of not changing anything.


```python
events = ['login fail', 'login fail', 'login', 'address change', 'buy', 'logout']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))
posterior = analyse_events(prior, events, legal_actions)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * posterior))
```

    Given the evidences 'login fail,login fail,login,address change,buy,logout', what is the posterior probability that our user is a buyer?
    	* "login fail" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "login fail" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "login" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "address change" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "buy" is positive. Prior probability is: 50.00%. Updated probability is: 60.00%
    	* "logout" is unknown. Prior probability is: 60.00%. Updated probability is: 60.00%
    	* "search" is negative. Prior probability is: 60.00%. Updated probability is: 46.15%
    	* "sell" is negative. Prior probability is: 46.15%. Updated probability is: 61.96%
    	* "view" is negative. Prior probability is: 61.96%. Updated probability is: 52.05%
    Probability that our user is a buyer is: 52.05%


The presence of unknown variables should be a huge red flag. That user is doing something that isn't covered by our model. 

We could just ignore these variable because we might think that they are noise, or maybe we could create another probability distribution that addresses these new variables. Doing so however will require an exponential number of values.
