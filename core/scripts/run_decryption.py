from core.models.mcmc import MCMCDecrypter
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # read text files into variables
    with open("../data/war_and_peace.txt", "r", encoding="utf-8") as file:
        war_and_peace = file.read()
    with open("../data/symbols.txt", "r", encoding="utf-8") as file:
        symbols = file.read()
    with open("../data/message.txt", "r", encoding="utf-8") as file:
        message = file.read()

    # run decrypter
    mcmc = MCMCDecrypter(war_and_peace, symbols, message)

    # plot the log-likelihood
    log_likelihood = mcmc.log_likelihoods
    plt.figure()
    plt.plot(np.arange(len(log_likelihood)), log_likelihood)
    plt.title("Log-Likelihood")
    plt.show()