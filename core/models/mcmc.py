from collections import Counter
import numpy as np
from copy import copy
from core.models.distribution import StationaryDistribution, Transitions


def decrypt(message: str, sigma: dict):
    """ Assume sigma maps from letter -> encryption. Return the decryption. """
    sigma_inverse = {val: key for key, val in sigma.items()}
    return ''.join([sigma_inverse[char] for char in message])


def calculate_log_joint(phi: dict, psi: dict, message: str, sigma: dict):
    """
        1. From message and sigma find current decryption
        2. Get s_1
        3. Calculate the log joint according to
            log phi[s_1] + sum( log psi[s_t, s_{t-1}] )
        2. Find phi[s_1] as the stationary probability of first entry in message
        3. Find psi.at[s_t, s_t-1] for all 2 >= t >= N.
        4. Calculate product as above

    """

    epsilon = 1e-88

    # get 'decrypted' message to find s_i
    decrypted = decrypt(message, sigma)
    s_1 = decrypted[0]

    # this is phi(s_1) but we will calculate product as we go so call it joint
    log_joint = np.log(phi.get(s_1))

    # find all transition log probabilities
    for t in range(1, len(decrypted)):

        # get s_t and s_t-1
        s_t = decrypted[t]
        s_t_minus_1 = decrypted[t-1]

        # find psi at pos t
        psi_t = psi[s_t_minus_1].get(s_t, epsilon)

        log_joint = log_joint + np.log(psi_t)

    return log_joint


def accept(message: str, sigma: dict, sigma_prime: dict, phi: dict, psi: dict):
    """
        1. Calculate the log joint probabilities of the message given sigma and sigma_prime
        2. Calculate Acceptance Probability
        3. Generate uniform random number
        4. Accept or reject
    """

    sigma_log_joint = calculate_log_joint(phi, psi, message, sigma)
    sigma_prime_log_joint = calculate_log_joint(phi, psi, message, sigma_prime)

    # acceptance probability according to log min equation above (np.log(1) =0 )
    acceptance_prob = min(0, sigma_prime_log_joint - sigma_log_joint)

    # random draw
    p = np.random.rand()
    log_p = np.log(p)

    # accept or reject
    return log_p < acceptance_prob, sigma_log_joint


def generate_proposal(sigma: dict, symbols: str):
    """ Swap the mappings of two characters randomly (without replacement) """

    # do not want inplace changes
    sigma_prime = copy(sigma)

    # choose 2 random chars
    index_1, index_2 = np.random.choice(np.arange(len(symbols)), 2, replace=False)
    symbol_1, symbol_2 = symbols[index_1], symbols[index_2]

    # swap mappings
    sigma_prime[symbol_1], sigma_prime[symbol_2] = sigma_prime[symbol_2], sigma_prime[symbol_1]
    return sigma_prime


class MCMCDecrypter:
    """

    Use MCMC to decrypt the message
    Store:
        phi: stationary distribution
        psi: transition matrix
        valid_symbols: the symbols that can occur in the message
        max_iter: run until max iter to prevent me needing to Cntrl C
        decryption: the decrypted message (at the current stage)
        log_likelihoods: log-likelihood at each step
        sigma_0: initial sigma

    """

    def __init__(self, text: str, valid_symbols: str, encrypted_message: str, max_iter: int = 25000):
        self.phi = StationaryDistribution(text=text, valid_symbols=valid_symbols)
        self.psi = Transitions(text=text, valid_symbols=valid_symbols)
        self.valid_symbols = self.psi.valid_symbols
        self.max_iter = max_iter
        self.decryption = ""

        self.encrypted_message = encrypted_message
        self.sigma_0 = self._init_sigma()
        self.sigma_prime = None

        self.log_likelihoods = []
        self.run()

    def _init_sigma(self):
        """

        Initialise Sigma by mapping the most frequent characters in the message to the characters with the respective
            highest stationary probabilities.

        :return:
        """
        phi_most_common = [tup[0] for tup in Counter(self.phi.distribution).most_common()]
        encrypt_most_common = [tup[0] for tup in Counter(self.encrypted_message).most_common()]

        sigma_0 = {}
        for i in range(len(encrypt_most_common)):
            sigma_0[phi_most_common[i]] = encrypt_most_common[i]

        real_not_mapped = ''.join([char for char in self.valid_symbols if char not in sigma_0.keys()])
        encrypt_not_mapped = ''.join([char for char in self.valid_symbols if char not in sigma_0.values()])
        for i in range(len(real_not_mapped)):
            sigma_0[real_not_mapped[i]] = encrypt_not_mapped[i]

        return sigma_0

    def run(self):
        """ Run MCMC """

        i = 0
        sigma = copy(self.sigma_0)

        while i < self.max_iter:
            # generate proposal
            sigma_prime = generate_proposal(sigma, self.valid_symbols)

            # accept or reject sigma
            _accept, log_like = accept(self.encrypted_message, sigma, sigma_prime, self.phi.distribution, self.psi.transitions)
            self.log_likelihoods.append(log_like)

            if _accept:
                sigma = sigma_prime

            # print first 60 letters every 100 iterations
            if not i % 100:
                self.decryption = decrypt(self.encrypted_message[:60], sigma)
                print(self.decryption)

            i += 1  # increment iteration

        # store the last sigma
        self.sigma_prime = sigma




