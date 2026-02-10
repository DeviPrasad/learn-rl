# The Reference Text Book

Deep Reinforcement Learning Hands-On. Third Edition, Packt Publishing, 2024.
- Maxim Lapan.

## The Code Repository of the Book

https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition


## Chapter 2. OpenAI Gym API and Gymnasium

The official web site for learning Gymnasium API is
https://gymnasium.farama.org/index.html.

Gymnasium offers
- uniform API for an RL agent and
- a number of RL environments.

I think it is a good idea to simply peruse the documentation. You will see a handful of tutorials and API docs providing adequate introduction.

## The First Step
Try reading and running the code described in the **Basic Usage** section https://gymnasium.farama.org/introduction/basic_usage/


## Creating a Toy Custom Environment
Take a look at the documentation for creating new environments: https://gymnasium.farama.org/introduction/create_custom_env/

We will develop/write similar code without using *numpy*. We will use only the built-in functions of Python and write wrappers and helper functions when required (to suite the abstractions).

The aim of this exercise is to understand the structure of the API and to develop **confidence** that we can build more involved environments.

## A Side Exercise
In order to keep things simple, you could try to define the environment for the **Recycling Robot** we have discussed in the classroom.

## The Env API
Keep the documentation for the *Env* API open for reference: https://gymnasium.farama.org/api/env/

## Environment Design Tips
https://gymnasium.farama.org/introduction/create_custom_env/#real-world-environment-design-tips

### Initializing the Environment
Try to understand how to derive a new environment from: https://gymnasium.farama.org/introduction/create_custom_env/#environment-init

#### Note
While trying to build this you should list all abstractions the documentation mentions as well as uses. This will help you in understanding how libraries are designed for generic use.

Are you comfortable with the nouns used in the text?

    - Action space
    - Observation


 Could you navigate around and get a grasp of the following abstraction?

    - Box space
