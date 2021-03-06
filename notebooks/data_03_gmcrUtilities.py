# Copyright:   (c) Oskar Petersons 2013

"""Assorted utility functions required by GMCR-py modules."""

import itertools
import numpy

bitFlip = {'N': 'Y', 'Y': 'N'}


def reducePatterns(patterns):
    """Reduce patterns into compact dash notation.

    Effectively a partial implementation of the Quine-McCluskey Algorithm?
    """
    for p in patterns:
        if len(p) != len(patterns[0]):
            raise ValueError("Patterns have different lengths.")

    if type(patterns) is not list:
        raise TypeError("Patterns must be provided as a list.")

    newPatterns = []
    matched = []
    for x, p1 in enumerate(patterns):
        if x in matched:
            continue
        for y, p2 in enumerate(patterns[x + 1:], 1):
            if x + y in matched:
                continue
            diffs = 0
            for idx, bit in enumerate(zip(p1, p2)):
                if bit[0] != bit[1]:
                    diffs += 1
                    dbit = idx
                if diffs > 1:
                    break
            if diffs == 1:
                newPatterns.append(p1[:dbit] + '-' + p1[dbit + 1:])
                matched += [x, x + y]
                break
        if x not in matched:
            newPatterns.append(p1)
    if matched:
        newPatterns = reducePatterns(newPatterns)
    return newPatterns

def define_moves(nr_decisions, nr_players):
    """Only to for situations where number of decisions for each player is the same"""
    import itertools

    nr_decisions = int(nr_decisions)

    arr = '1' + (int(nr_decisions)-1)*'0'
    all_options = [''.join(row) for row in numpy.array(list(itertools.permutations(arr,nr_decisions)))]
    feasibles = [''.join(row) for row in set(list(itertools.product(all_options, repeat = nr_players)))]

    return feasibles

def expandPatterns(patterns):
    """Expand patterns so that they contain no dashes."""
    newPatterns = []

    # for pat in patterns:
    #     if '-' in pat:
    #         adding = [pat.replace('-', 'Y', 1), pat.replace('-', 'N', 1)]
    #         newPatterns += expandPatterns(adding)
    #     else:
    #         newPatterns += [pat]
    
    return newPatterns


def yn2dec(ynState):
    """Convert a binary YN string into a decimal number."""
    bit = 0
    output = 0
    for m in ynState:
        if m == '1':
            output += 2**bit
        bit += 1
    return output


def dec2yn(decState, numOpts):
    """Convert a decimal number into a binary string of appropriate length."""
    output = bin(decState).lstrip("0b").zfill(
        numOpts)[::-1]#.replace('1', 'Y').replace('0', 'N')
    return output

def find_name(state, nr_decisions, payoffs, baseline):
    #print(payoffs, "payoffs")
    #print(baseline, "baseline")
    if numpy.all(numpy.diff(numpy.where(numpy.array(list(state))=='1'))==nr_decisions):
        #if numpy.where(numpy.array(list(state))=='1')[0][0] != baseline:
        policy_index = numpy.where(numpy.array(list(state))=='1')[0][0]
        name = payoffs.index[policy_index]
        #print(name, "policy name")
        return name #numpy.where(numpy.array(list(state))=='1')[0][0]
    else:
        return baseline


def _subtractPattern(feas, sub):
    """Remove infeasible condition 'sub' from feasible condition 'feas'."""
    if len(feas) != len(sub):
        raise ValueError("Patterns have different lengths.")
    sub = [x for x in enumerate(sub) if x[1] != '-']
    # check if targ overlaps with state:
    for x in sub:
        idx, val = x
        if feas[idx] == bitFlip[val]:  # if no overlap then no change,
            return [feas]              # and return the same feasible condition
    # subtract overlap if it exists
    remainingStates = []
    curr = feas
    for x in sub:
        idx, val = x
        if curr[idx] == '-':
            remainingStates.append(curr[:idx] + bitFlip[val] + curr[idx + 1:])
            curr = curr[:idx] + val + curr[idx + 1:]
    return remainingStates


def rmvSt(feas, rmv):
    """Subtract YND 'rmv' from states list of states 'feas'.

    feas: list of YND states.
    rmv: a single YND state.
    returns: list feas - rmv, and the number of states removed.
    """
    orig = sum([2**x.count('-') for x in feas])  # Original number of states
    newfeas = []
    for pattern in feas:
        newfeas += _subtractPattern(pattern, rmv)
    newfeas = reducePatterns(newfeas)
    # number of states removed
    numRmvd = orig - sum([2**x.count('-') for x in newfeas])
    return newfeas, numRmvd


def subtractStateSets(originalStates, statesToRemove):
    """Return the originalStates minus the statesToRemove.

    originalStates: list of YND states.
    statesToRemove: list of YND states.
    """
    newStates = reducePatterns(originalStates)
    for rmv in reducePatterns(statesToRemove):
        newStates = rmvSt(newStates, rmv)[0]

    return newStates


def mutuallyExclusive(mutEx):
    """Return the equivalent set of infeasible states."""
    return list(itertools.combinations(mutEx, 2))


def orderedNumbers(decimalList):
    """Create translation dictionaries for using ordered numbers.

    Generates the decimal->ordered and ordered->decimal translation
    dictionaries for a list of decimal values.
    """
    toOrdered = {}        # decimal -> ordered dictionary
    toDecimal = {}         # ordered -> decimal dictionary
    for i, x in enumerate(decimalList, 1):
        toOrdered[x] = i
        toDecimal[i] = x
    return toOrdered, toDecimal


def validatePreferenceRanking(prefRank, feasibles):
    """Check that the preference ranking given is valid."""
    alreadySeen = []
    if not isinstance(prefRank, list):
        return "Invalid format."
    for state in prefRank:
        if state in feasibles.ordered:
            if state in alreadySeen:
                return "State {} cannot appear more than once.".format(state)
            alreadySeen.append(state)
        else:
            try:
                for subSt in state:
                    if subSt in feasibles.ordered:
                        if subSt in alreadySeen:
                            return ("State {} cannot appear more than"
                                    " once.").format(subSt)
                        alreadySeen.append(subSt)
                    else:
                        return ("State {} is not a feasible "
                                "state.").format(subSt)
            except TypeError:
                return "State {} is not a feasible state".format(state)

    for state in feasibles.ordered:
        if state not in alreadySeen:
            return "State {} is missing.".format(state)

    return None


def mapPrefRank2Payoffs(preferenceRanking, feasibles, feasibles_names, options, name, actual_payoffs):
    ## SS: Modified this
    """Map the preference rankings into payoff values for each state."""
    #print(actual_payoffs)
    payoffs = numpy.zeros(len(feasibles))    # clean payoffs array
    #print(len(feasibles), "feasibles length")
    # SS: Used to be state -1, replaced with state given the payoff rankings also start at 0
    # use position in preference ranking to give a payoff value.
    for idx, state in enumerate(preferenceRanking):
        try:
            for subState in state:
                if feasibles_names[subState] == 'baseline':
                    #print(actual_payoffs[[name]].loc[3])
                    payoffs[subState] = 2**(actual_payoffs[[name]].iloc[-1])
                    #print(payoffs[subState], "baseline")
                else:
                    #print(actual_payoffs[[name]].loc[feasibles_names[subState]])

                    payoffs[subState] = 2**(actual_payoffs[[name]].loc[feasibles_names[subState]])
                    #print(actual_payoffs[[name]].loc[feasibles_names[subState]])
                    #print(payoffs[subState])
                #payoffs[subState] = len(feasibles) - idx
        except TypeError:
            payoffs[state] = len(feasibles) - idx

    #print(payoffs)

    if 0 in payoffs:
        state = feasibles.ordered[payoffs.index(0)]
        raise Exception(("Feasible state '%s' for DM was not included in the "
                         "preference ranking").format(state))

    return payoffs


def prefPriorities2payoffs(preferences, feasibles):
    """Rank the states for a DM, generating payoff values.

    Ranking is based on Preference Prioritization, and output payoff values
    are sequential.
    """
    # generate initial payoffs
    payoffsRaw = numpy.zeros(len(feasibles), numpy.int_)
    for preference in preferences:
        for state in feasibles.decimal:
            if preference.test(state):
                payoffsRaw[feasibles.toOrdered[state] - 1] += preference.weight

    # Reduce magnitude of payoffs.
    # Do not do this if weights had special meaning.
    uniquePayoffs = numpy.unique(payoffsRaw)
    preferenceRanking = []
    payoffs = payoffsRaw.copy()  # creates a copy

    for idx, value in enumerate(uniquePayoffs):
        for jdx, pay in enumerate(payoffsRaw):
            if pay == value:
                payoffs[jdx] = idx + 1
        stateSet = [idx + 1 for idx, pay in enumerate(payoffsRaw)
                    if pay == value]
        if len(stateSet) > 1:
            preferenceRanking.append(stateSet)
        else:
            preferenceRanking.append(stateSet[0])

    # necessary to put most preferred states at beginning instead of end
    preferenceRanking.reverse()

    return payoffs, preferenceRanking
