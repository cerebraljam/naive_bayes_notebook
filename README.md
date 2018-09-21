
# Naive Bayes - Updating a Posterior Probability From A List of Events

The idea of this notebook is to demonstrate how a list of actions can be used to update the posterior probabilities of something being true or not.

## Scenario
As an example scenario, Let's say that we are monitoring a web application, and we want to identify if a user is a buyer (1) or something else (0).

## Observable Variables
To keep this simple at first, I will limit the possible actions to 4 actions: search, sell, buy, view. 

Regarding the probability distributions, the "pgmpy" library offers built in functionalities to self train a probability distributions from a list of events, but for education purposes, we will guestimate it here.



```python
# Defining the tests and their probabilities

buyer_likelihood = {}

buyer_likelihood['search'] = { # likelihood that a buyer will use the search function
    1 : { 
        1: 0.6, # P(buyer | search) (True Positive)
        0: 0.3 # P(-buyer | search) (False Positive)
    },
    0: {
        1: 0.4, # P(buyer | -search) (True Negative)
        0: 0.7 # P(-buyer | -search) (False Negative)
    }
}

buyer_likelihood['sell'] = { # likelihoods that a buyer will sell an item
    1 : {
        1: 0.05, # P(buyer | sell) (True Positive)
        0: 0.5 # P(-buyer | sell) (False Positive)
    },
    0: {
        1: 0.95, # P(buyer | -sell) (True Negative)
        0: 0.5 # P(-buyer | -sell) (False Negative)
    }
}

buyer_likelihood['buy'] = { # likelihoods that a buyer will buy an item
    1 : {
        1: 0.3, # P(buyer | buy) (True Positive)
        0: 0.2 # P(-buyer | buy) (False Positive)
    },
    0: {
        1: 0.7, # P(buyer | -buy) (True Negative)
        0: 0.8 # P(-buyer | -buy) (False Negative)
    }
}

buyer_likelihood['view'] = { # likelihoods that a buyer will view the details of an item
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
def analyse_events(prior, events, legals, likelihood):
    posterior = prior
    for ee in events:
        posterior = update_probability(posterior, ee, likelihood, 1)
    for ll in legals:
        if ll not in events:
            posterior = update_probability(posterior, ll, likelihood, 0)
    return posterior
```


```python
# Prior: What is our initial belief that the user is a buyer
prior = 0.5 

# List of legal actions
buyer_legal_actions = buyer_likelihood.keys()
```

# Updating Our Bliefs, Given Evidences

Let's say that we are observing a chain of events. For each of these events, we want to update our belief about the user.

## Buyer Profile


```python
events = ['search', 'view', 'search', 'view', 'view', 'buy']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))

buyer_posterior = analyse_events(prior, events, buyer_legal_actions, buyer_likelihood)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * buyer_posterior))
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


## Seller Profile

Notice that our test is trying to identify if the user is a buyer, not if s/he is a seller.
If we wanted to know the probability that the user is a seller, we would need different likelihood data.

Still, let's see how this model reacts.


```python
events = ['search', 'view', 'view', 'view', 'sell', 'sell']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))

buyer_posterior = analyse_events(prior, events, buyer_legal_actions, buyer_likelihood)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * buyer_posterior))
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


## Power "whatever" Profile

Let's say that our user is actually a hobbyist trying to buy cheap and resell at a higher price, and therefore make money over other sellers who are willing to sell for cheaper than they should. How would our model react?


```python
events = ['buy', 'sell', 'view', 'sell', 'buy', 'view']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))

buyer_posterior = analyse_events(prior, events, buyer_legal_actions, buyer_likelihood)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * buyer_posterior))
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

## What Happen If We Test With Unknown Events?

Let's say that our user is trying to commit fraud, what would happen?

Because the variables are unknown, we can't update positively or negatively the posterior. We could of course configure a default value that would move the results in one direction or the other, but this could also generate false results.

Because of this, the update_probability function will use a 50%/50% probability, which has the impact of not changing anything.


```python
events = ['login fail', 'login fail', 'login', 'address change', 'buy', 'logout']

print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))

buyer_posterior = analyse_events(prior, events, buyer_legal_actions, buyer_likelihood)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * buyer_posterior))
```

    Given the evidences 'login fail,login fail,login,address change,buy,logout', what is the posterior probability that our user is a buyer?
    	* "login fail" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "login fail" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "login" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "address change" is unknown. Prior probability is: 50.00%. Updated probability is: 50.00%
    	* "buy" is positive. Prior probability is: 50.00%. Updated probability is: 60.00%
    	* "logout" is unknown. Prior probability is: 60.00%. Updated probability is: 60.00%
    	* "view" is negative. Prior probability is: 60.00%. Updated probability is: 50.00%
    	* "sell" is negative. Prior probability is: 50.00%. Updated probability is: 65.52%
    	* "search" is negative. Prior probability is: 65.52%. Updated probability is: 52.05%
    Probability that our user is a buyer is: 52.05%


The presence of unknown variables should be a huge red flag that user is doing something that isn't covered by our model. 

We could just ignore these variable because we might think that they are noise, or maybe we could create another probability distribution that addresses these new variables.

### Add The Missing Variables To The Buyer's Likelihood

Let's add the missing variables for the buyer and try again.


```python
buyer_likelihood['login'] = { # likelihood that a fraudster will login
    1 : { 
        1: 0.8, # P(fraudster | login) (True Positive)
        0: 0.6 # P(-fraudster | login) (False Positive)
    },
    0: {
        1: 0.2, # P(fraudster | -login) (True Negative)
        0: 0.4 # P(-fraudster | -login) (False Negative)
    }
}

buyer_likelihood['login fail'] = { # likelihood that a fraudster will fail to login
    1 : { 
        1: 0.2, # P(fraudster | login fail) (True Positive)
        0: 0.9 # P(-fraudster | login fail) (False Positive)
    },
    0: {
        1: 0.8, # P(fraudster | -login fail) (True Negative)
        0: 0.1 # P(-fraudster | -login fail) (False Negative)
    }
}

buyer_likelihood['address change'] = { # likelihood that a fraudster will use the address change function
    1 : { 
        1: 0.4, # P(fraudster | address change) (True Positive)
        0: 0.8 # P(-fraudster | address change) (False Positive)
    },
    0: {
        1: 0.6, # P(fraudster | -address change) (True Negative)
        0: 0.2 # P(-fraudster | -address change) (False Negative)
    }
}

buyer_likelihood['logout'] = { # likelihood that a fraudster will use the logout function
    1 : { 
        1: 0.2, # P(fraudster | logout) (True Positive)
        0: 0.7 # P(-fraudster | logout) (False Positive)
    },
    0: {
        1: 0.8, # P(fraudster | -logout) (True Negative)
        0: 0.3 # P(-fraudster | -logout) (False Negative)
    }
}

```


```python
print("Given the evidences '{}', what is the posterior probability that our user is a buyer?".format(",".join(events)))

buyer_posterior = analyse_events(prior, events, buyer_legal_actions, buyer_likelihood)
print("Probability that our user is a buyer is: {:.2f}%".format(100 * buyer_posterior))
```

    Given the evidences 'login fail,login fail,login,address change,buy,logout', what is the posterior probability that our user is a buyer?
    	* "login fail" is positive. Prior probability is: 50.00%. Updated probability is: 18.18%
    	* "login fail" is positive. Prior probability is: 18.18%. Updated probability is: 4.71%
    	* "login" is positive. Prior probability is: 4.71%. Updated probability is: 6.18%
    	* "address change" is positive. Prior probability is: 6.18%. Updated probability is: 3.19%
    	* "buy" is positive. Prior probability is: 3.19%. Updated probability is: 4.71%
    	* "logout" is positive. Prior probability is: 4.71%. Updated probability is: 1.39%
    	* "view" is negative. Prior probability is: 1.39%. Updated probability is: 0.93%
    	* "sell" is negative. Prior probability is: 0.93%. Updated probability is: 1.76%
    	* "search" is negative. Prior probability is: 1.76%. Updated probability is: 1.01%
    Probability that our user is a buyer is: 1.01%


## Probability Distributions For Fraudsters

After observing our profiles and logs, and investigating the behaviour that our fraudster did, we can create a probability distribution that would make it easier for us to spot it again.


```python
# Here we define probabilities distribution for a fraudsters

fraudster_likelihood = {}

fraudster_likelihood['login'] = { # likelihood that a fraudster will login
    1 : { 
        1: 0.6, # P(fraudster | login) (True Positive)
        0: 0.8 # P(-fraudster | login) (False Positive)
    },
    0: {
        1: 0.4, # P(fraudster | -login) (True Negative)
        0: 0.2 # P(-fraudster | -login) (False Negative)
    }
}

fraudster_likelihood['login fail'] = { # likelihood that a fraudster will fail to login
    1 : { 
        1: 0.55, # P(fraudster | login fail) (True Positive)
        0: 0.2 # P(-fraudster | login fail) (False Positive)
    },
    0: {
        1: 0.45, # P(fraudster | -login fail) (True Negative)
        0: 0.8 # P(-fraudster | -login fail) (False Negative)
    }
}

fraudster_likelihood['address change'] = { # likelihood that a fraudster will use the address change function
    1 : { 
        1: 0.8, # P(fraudster | address change) (True Positive)
        0: 0.4 # P(-fraudster | address change) (False Positive)
    },
    0: {
        1: 0.2, # P(fraudster | -address change) (True Negative)
        0: 0.6 # P(-fraudster | -address change) (False Negative)
    }
}

fraudster_likelihood['logout'] = { # likelihood that a fraudster will use the logout function
    1 : { 
        1: 0.7, # P(fraudster | logout) (True Positive)
        0: 0.2 # P(-fraudster | logout) (False Positive)
    },
    0: {
        1: 0.3, # P(fraudster | -logout) (True Negative)
        0: 0.8 # P(-fraudster | -logout) (False Negative)
    }
}

fraudster_likelihood['search'] = { # likelihood that a fraudster will use the search function
    1 : { 
        1: 0.8, # P(fraudster | search) (True Positive)
        0: 0.6 # P(-fraudster | search) (False Positive)
    },
    0: {
        1: 0.2, # P(fraudster | -search) (True Negative)
        0: 0.4 # P(-fraudster | -search) (False Negative)
    }
}

fraudster_likelihood['sell'] = { # likelihoods that a fraudster will sell an item
    1 : {
        1: 0.05, # P(fraudster | sell) (True Positive)
        0: 0.5 # P(-fraudster | sell) (False Positive)
    },
    0: {
        1: 0.95, # P(fraudster | -sell) (True Negative)
        0: 0.5 # P(-fraudster | -sell) (False Negative)
    }
}

fraudster_likelihood['buy'] = { # likelihoods that a fraudster will buy an item
    1 : {
        1: 0.4, # P(fraudster | buy) (True Positive)
        0: 0.8 # P(-fraudster | buy) (False Positive)
    },
    0: {
        1: 0.6, # P(fraudster | -buy) (True Negative)
        0: 0.2 # P(-fraudster | -buy) (False Negative)
    }
}

fraudster_likelihood['view'] = { # likelihoods that a fraudster will view the details of an item
    1 : {
        1: 0.4, # P(fraudster | view) (True Positive)
        0: 0.85 # P(-fraudster | view) (False Positive)
    },
    0: {
        1: 0.6, # P(fraudster | -view) (True Negative)
        0: 0.15 # P(-fraudster | -view) (False Negative)
    }
}
```

In prevision that we might want to add more profile, let's put them together in a single dictionary.


```python
likelihood = {}

likelihood['buyer'] = buyer_likelihood
likelihood['fraudster'] = fraudster_likelihood
```

### Evaluating the Posterior Probability Of A List of Events Matching A Buyer Or Fraudster Profile


```python
events = ['login fail', 'login fail', 'login', 'address change', 'buy', 'logout']

print("Given the evidences '{}'...".format(",".join(events)))
for user,distribution in likelihood.items():
    print("what is the posterior probability that our user is a {}?".format(user))

    # List of legal actions
    legal_actions = likelihood[user].keys()
    posterior = analyse_events(prior, events, legal_actions, likelihood[user])
    print("Probability that our user is a {} is: {:.2f}%\n".format(user, 100 * posterior))
```

    Given the evidences 'login fail,login fail,login,address change,buy,logout'...
    what is the posterior probability that our user is a buyer?
    	* "login fail" is positive. Prior probability is: 50.00%. Updated probability is: 18.18%
    	* "login fail" is positive. Prior probability is: 18.18%. Updated probability is: 4.71%
    	* "login" is positive. Prior probability is: 4.71%. Updated probability is: 6.18%
    	* "address change" is positive. Prior probability is: 6.18%. Updated probability is: 3.19%
    	* "buy" is positive. Prior probability is: 3.19%. Updated probability is: 4.71%
    	* "logout" is positive. Prior probability is: 4.71%. Updated probability is: 1.39%
    	* "view" is negative. Prior probability is: 1.39%. Updated probability is: 0.93%
    	* "sell" is negative. Prior probability is: 0.93%. Updated probability is: 1.76%
    	* "search" is negative. Prior probability is: 1.76%. Updated probability is: 1.01%
    Probability that our user is a buyer is: 1.01%
    
    what is the posterior probability that our user is a fraudster?
    	* "login fail" is positive. Prior probability is: 50.00%. Updated probability is: 73.33%
    	* "login fail" is positive. Prior probability is: 73.33%. Updated probability is: 88.32%
    	* "login" is positive. Prior probability is: 88.32%. Updated probability is: 85.01%
    	* "address change" is positive. Prior probability is: 85.01%. Updated probability is: 91.90%
    	* "buy" is positive. Prior probability is: 91.90%. Updated probability is: 85.01%
    	* "logout" is positive. Prior probability is: 85.01%. Updated probability is: 95.20%
    	* "view" is negative. Prior probability is: 95.20%. Updated probability is: 98.76%
    	* "sell" is negative. Prior probability is: 98.76%. Updated probability is: 99.34%
    	* "search" is negative. Prior probability is: 99.34%. Updated probability is: 98.69%
    Probability that our user is a fraudster is: 98.69%
    

