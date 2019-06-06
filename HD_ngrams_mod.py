#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:13:43 2019

@author: denkle
"""
import numpy as np
import time

def ngram_encode_mod(str, HD_aphabet, aphabet, n_size): # method for mapping n-gram statistics of a word to an N-dimensional HD vector
    HD_ngram = np.zeros(HD_aphabet.shape[1]) # will store n-gram statistics mapped to HD vector
    full_str = '#' + str + '#' # include extra symbols to the string
    shift=n_size-1
        
    # form the vector for the first n-gram
    hdgram = HD_aphabet[aphabet.find(full_str[0]), :] # picks HD vector for the first symbol in the current n-gram
    for ng in range(1, n_size): #loops through the rest of symbols in the current n-gram
        hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[0+ng]), :], ng) # two operations simultaneously; binding via elementvise multiplication; rotation via cyclic shift        
    HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram
        
    for l in range(1, len(full_str)-shift):  # form all other n-grams using the HD vector for the first one      
        hdgram = np.roll( hdgram * HD_aphabet[aphabet.find(full_str[l-1]), :], -1) * np.roll(HD_aphabet[aphabet.find(full_str[l+shift]), :], shift) # improved implementation of forming HD vectors for n-grams
        HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram
    
    HD_ngram_norm = np.sqrt(HD_aphabet.shape[1]) * (HD_ngram/ np.linalg.norm(HD_ngram) )  # normalizes HD-vector so that its norm equals sqrt(N)       
    return HD_ngram_norm # output normalized HD mapping



def ngram_encode_hybrid(str, HD_aphabet, aphabet, n_size): # method for mapping n-gram statistics of a word to an N-dimensional HD vector
    HD_ngram = np.zeros(HD_aphabet.shape[1]) # will store n-gram statistics mapped to HD vector
    full_str = '#' + str + '#' # include extra symbols to the string
    shift=n_size-1
    
    if n_size < 3: # use the initial implementation
        #adjust the string for n-gram size
        if n_size == 1:
            full_str_e=full_str            
        else:
            full_str_e=full_str[:-(n_size-1)]    
            
        for il, l in enumerate(full_str_e): # loops through all n-grams
            hdgram = HD_aphabet[aphabet.find(full_str[il]), :] # picks HD vector for the first symbol in the current n-gram
            
            for ng in range(1, n_size): #loops through the rest of symbols in the current n-gram
                    hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[il+ng]), :], ng) # two operations simultaneously; binding via elementvise multiplication; rotation via cyclic shift
    
            HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram

    else:    # use the modified implementation

        # form the vector for the first n-gram
        hdgram = HD_aphabet[aphabet.find(full_str[0]), :] # picks HD vector for the first symbol in the current n-gram
        for ng in range(1, n_size): #loops through the rest of symbols in the current n-gram
            hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[0+ng]), :], ng) # two operations simultaneously; binding via elementvise multiplication; rotation via cyclic shift        
        HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram
            
        for l in range(1, len(full_str)-shift): # form all other n-grams using the HD vector for the first one    
            hdgram = np.roll( hdgram * HD_aphabet[aphabet.find(full_str[l-1]), :], -1) * np.roll(HD_aphabet[aphabet.find(full_str[l+shift]), :], shift) # improved implementation of forming HD vectors for n-grams
            HD_ngram += hdgram # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram
    
    HD_ngram_norm = np.sqrt(HD_aphabet.shape[1]) * (HD_ngram/ np.linalg.norm(HD_ngram) )  # normalizes HD-vector so that its norm equals sqrt(N)       
    
    return HD_ngram_norm # output normalized HD mapping



N = 1000 # set the desired dimensionality of HD vectors
n_size=2 # n-gram size
aphabet = 'abcdefghijklmnopqrstuvwxyz# ' #fix the alphabet. Note, we assume that capital letters are not in use 
np.random.seed(1) # for reproducibility
HD_aphabet = 2 * (np.random.randn(len(aphabet), N) < 0) - 1 # generates bipolar {-1, +1}^N HD vectors; one random HD vector per symbol in the alphabet

#str='jump' # example string to represent using n-grams
str='intrigued started to work it out the third installment of the twilight saga eclipse proved to be a small stepup from the first two movies lowrie atoned for an error he made in the top half of the inning when he dropped a foul popup while playing first base but she couldnt sleep knowing that her mother was sitting alone somewhere in a big foreign airport that had gone into crisis mode call the minot at to reserve your spot for beginning duplicate bridge monday evenings plucking the track from her am sasha fierce beyonces halo has become one of her signature songs wow sound smart should change his name to really dumb it dabbles in several facets of the financial services industry from its jpmorgan investment banking chase credit cards and various banking operations the developing industrial applications for silver are exciting and expect longterm growth here rick morgan has been fishing lower manitou since he was a kid growing up in hibbing in the late he may not be a good guy but hes not a rapist at least not yet we have taken this request under consideration in gardner was facing trial for killing melvyn otterstrom a bartender in salt lake city when a girlfriend slipped him a gun at the courthouse soon after more offers came piling in from other schools central michigan miami western michigan and ohio for the current crop of condo dwellers losing the convenience of a car is part of the price of home ownership by values were in retreat will julian crocker look into this small problem but she said that officer didnt want to hear any more these were on increases in gross and average from their pathetic predecessors though neither measure was ever an accurate barometer of industry health the participants bacon cheeseburger patty with bacon lettuce tomato and mayo phillies catcher carlos ruiz connects for a gamewinning home run in the inning against the cardinals tango don cheadle is an undercover cop posing as a drug lord watching his criminal compatriot wesley snipes in a restrained return to form return to power after a lengthy prison stretch we are eliminating sales and management company cars altogether and moving those positions to an allowance said lisa kneggs fleet manager on a per school dayhourly basis has anyone done the math on this why dont we just pay everyone not to work peacock asks referring to losing the streak lot depends on the pitching staff of course but as the smoke cleared lost became as twitter fan joe hewitt put it a soap opera not an intellectual scifi thriller and thus a waste of time that is something hill hopes they all take seriously in worlds recent tests on mobile bandwidth speeds ranks first hands down its too early to judge whether a large number of donors across the country will pour money into the race as they did for brown but browns appearance here is likely to increase that possibility as the contest gains notice president ahmadinejad further announced irans decision to postpone talks with the world powers until august in a move to punish the west weir fire escape extension burley bridge mills viaduct road burley hes a house republican guy he may have been wearing number on his hip but tonights reebok boston indoor games at the reggie lewis center here was anything but unlucky for twotime olympic medallist bernard lagat adam vinatieri answered with a bomb of his own from yards many towns around the state are facing tightening budgets as certain services and repairs are left for another day whatrsquos amazing about jayz is that even nearing an age unheard of for a viable rapper even a decade ago the artist keeps putting out genredefying and essential albums is it rugged enough to handle outside terrain the war is coming to their cities the chechen rebel leader said in an interview on an islamist website he lasted only innings for his secondshortest outing of ford motor co has extended buyout offers once again to workers at its hamburg stamping plant a sign that the automaker is pushing hard to reduce its production work force rich barber says thats as it should be card companies can slap a penalty interest rate on existing balances if a customer falls at least days behind on payments obama then took on jay leno the nights entertainment saying great to see you jay the order was made through ovhs automated systems on the eve of retirement minnesota duluth chancellor kathryn martin seeks to finish her transformation of campus you are now over limit and so an over limit fee of rs can also be applied making the owed balance on the first page youll get photos of white folks being beaten up by groups of blacks and arabs in the spring mcdaniels made brandon marshall and tony scheffler who werent shy about expressing criticism his new jay cutlers its a sad day when a collegeuniversity honours a politician whose claim to fame was the businessoriented common sense revolution over someone anyone whose life serveds to inspire students to celebrate education the change while not material was accounted for on a retrospective basis and more closely aligns the depreciation policies with those of the companys drilling rigs which are depreciated based on operating days lets go to henry its popup menu lists all the pcs in your house that have been prepared for remote controlling including the kitchen laptop but there were other places to take a dip and a handful of bajans were taking advantage as dozed horn already had worked out for teams such as the carolina panthers and the philadelphia eagles but his times were in the mediocre to range notre dame beat providence next at syracuse saturday evolution as a search engine for commensurate energy and nutrient niches was now possible evolution as a euros the survival of the most fitting on this or any other wet rocky and sunlit planet and in any galaxy taking legal and critical measures of control in the sphere on nuclear security the international community should not ignore the global trends in energy and high technologies the judges did not goof in picking the medal winners mostly because the athletes last night made their jobs so easy ryan cross or potentially digby iaoni seem logical to me to add a bit of size what are your thoughts susan joy share holds one of her fish puppets from the sleep of waters theatrical presentation at her animated library exhibit wednesday at out north the battery gives me about hours usage under normal web browsing together with some office applications it really pisses me off submitted by mikel on sat we should also stop taxing businesses as individuals but rather reduce rates to which would help business to grow and create jobs ousley also won titles in breakaway roping and bradley in tiedown roping beyond that im not sure anybody agrees on anything be ready for a bunch of hostiles they were kind of targeting him and if youre not giving your best effort it allows the other team to feed off of that troy orion tom of beclabito killed aug by an improvised explosives device when his squad went to the aid of another that had come under heavy fire how many did it during the era we had the answers designed for those who recently lost a loved one he would like to provide more coverage to the uninsured by allowing people to buy into medicaid coverage an unknown suspect shot into a the residence and a vehicle that was parked in front of craig st neither the vehicle nor the residence was occupied at the time backtoback home runs by dollar ridgell and jerry powell highlighted an eightrun firstinning as morehouse went on to a convincing second round victory ron mcclelland the pastpresident of the and law association sees the changes as a case of jobs and services being moved out of the community ryan wittman has the game is upsidedown ahh well the lower prices just give myself and you more time to get more gold before it moves much higher so take it as a blessing in disguise shes the best candidate weve had since shandy finnessey but better earth day events include an art exhibition featuring kimberly piazzas tire recycling concepts at the pacific pinball museum in alameda and a screening of natures half acre about insects and flowers at the walt disney family museum in san francisco most of the paratroopers are still arriving trying to assess conditions and find the right local officials to work with the most lush lowkey spot in the world with a code attached to it she really comes across as a hypocrite steve teques was fifth in the the anniversary earth day will be observed thursday and the mesa republic is taking a look at four local businesses that use green practices days a year harry and nancy plastered them up at every press conference health care is legislation and the supposed divide of polls are within the margin of error of said pollsin which case there is no provable divide and the use journalistically of the word divide is inappropriate they would show about as much understanding of our system reagan agreed immediately it is a known fact what we arent paying out in decent shelter and living allowances we are paying out later in medical expenses prisons and child protection services we enter like animals and go out like animals says samir sublaban a grocery store owner he has missed straight games since suffering a torn groin at ottawa on jan and isnt expected to return until after the olympic break theyre not viable in commercial theaters without movie stars at least the government will then give you a tax break so you can buy a house cousins is taller than jefferson but should be able to put up similar doubledouble numbers cant even seem to sign up on atts website to check that means the tests are set to ignore the majority of banks holdings of sovereign debt as they moved the riskier elements of the debt out of their trading books so that it was left out of the scope of the tests compelling story lines involving the city of new orleans and itsongoing recovery from hurricane katrina and the attempt at a secondsuper bowl ring for indianapolis quarterback peyton manningpropelled the viewership they lost their mental strength after qualifying for the world cup strike while the irons hot unite says another was not sent a form and while their lives dont revolve solely around potter these days its still a source of good fun lynn neuenswander public information officer for the department of behavioral health part of the countys continuum of care said the face of homelessness is changing so please leave well enough alone unless you can do better or help in some way he said once a month he plans to open both sides both party central and party zone for larger parties juan uribe walked and sanchez doubled home sandoval loan from a bank or credit union more financial institutions are offering shortterm loans to people with poor credit with the mailout of census questionnaires slightly more than a month away the census bureau will run three ads promoting census awareness during the super bowl telecast two during the pregame show and one during the third quarter she is saying she is doing this for fun and why cant moms have fun rookie percy harvin is a real threat as is visanthe shiancoe tds including playoffs how are we to judge the austerity measures just passed by the merkel government ms chithra would not be eliminated or replaced by anybody think ill miss the walks to the stadium more than the games themselves its dans understanding and we have neurological evidence that abbie is not capable of seeing or having any cognitive understanding of the children even if they were standing in front of her greene said predicts a percent increase in the number of marylanders traveling by car this weekend in trying to find his best diego led argentina into battle on occasions with a record of luftha'

start = time.time()
HD_ngram = ngram_encode_mod(str, HD_aphabet, aphabet, n_size) # HD_ngram is a projection of n-gram statistics for str to N-dimensional space. It can be used to learn the word embedding
end = time.time()
print(end - start) # get execution time

start = time.time()
HD_ngram2 = ngram_encode_hybrid(str, HD_aphabet, aphabet, n_size) # HD_ngram is a projection of n-gram statistics for str to N-dimensional space. It can be used to learn the word embedding
end = time.time()
print(end - start)  # get execution time

print(np.sum(np.abs(HD_ngram2-HD_ngram))) # check correctness of obtained HD vectors


