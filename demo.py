# dependencies.
import os
os.system('pip install clang==6.0.0.2')
os.system('pip install nltk==3.3')
os.system('pip install natsort')
import re
import gc
import time
import sys
import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import clang.cindex
import clang.enumerations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.metrics import accuracy_score
import pickle
from natsort import natsorted

# environment settings.
_COLAB_ = 0 if (os.getenv('COLAB_GPU', 'NONE') == 'NONE') else 1 # 0 : Local environment, 1 : Google Colaboratory.
# file paths.
rootPath = './drive/My Drive/Colab Notebooks/' if (_COLAB_) else './'
# dataPath = rootPath + '/data/'
# sDatPath = dataPath + '/security_patch/'
# pDatPath = dataPath + '/positives/'
# nDatPath = dataPath + '/negatives/'
testPath = rootPath + '/testdata/'
modlPath = rootPath + '/model/'
tempPath = rootPath + '/tmp/'
resuPath = rootPath + '/results/'
logsPath = rootPath + '/logs/'

# hyper-parameters. (affect GPU memory size)
_DiffEmbedDim_  = 128       # 128
_DiffMaxLen_    = 600       # 200(0.7), 314(0.8), 609(0.9), 1100(0.95), 2200(0.98), 3289(0.99), 5000(0.995), 10000(0.9997)
_DRnnHidSiz_    = 16        # 16
_MsgEmbedDim_   = 128       # 128
_MsgMaxLen_     = 200       # 54(0.9), 78(0.95), 130(0.98), 187(0.99), 268(0.995), 356(0.998), 516(0.999), 1434(1)
_MRnnHidSiz_    = 16        # 16
_TwinEmbedDim_  = 128       # 128
_TwinMaxLen_    = 800       # 224(0.8), 425(0.9), 755(0.95), 1448(0.98), 2270(0.99)
_TRnnHidSiz_    = 16        # 16
# hyper-parameters. (affect training speed)
_DRnnBatchSz_   = 128       # 128
_DRnnLearnRt_   = 0.0001    # 0.0001
_MRnnBatchSz_   = 128       # 128
_MRnnLearnRt_   = 0.0001    # 0.0001
_PRnnBatchSz_   = 256       # 256
_PRnnLearnRt_   = 0.0005    # 0.0005
_TRnnBatchSz_   = 256       # 256
_TRnnLearnRt_   = 0.0005    # 0.0005
# hyper-parameters. (trivial network parameters, unnecessary to modify)
_DiffExtraDim_  = 2         # 2
_TwinExtraDim_  = 1         # 1
_DRnnHidLay_    = 1         # 1
_MRnnHidLay_    = 1         # 1
_TRnnHidLay_    = 1         # 1
# hyper-parameters. (epoch related parameters, unnecessary to modify)
_DRnnMaxEpoch_  = 1000      # 1000
_DRnnPerEpoch_  = 1         # 1
_DRnnJudEpoch_  = 10        # 10
_MRnnMaxEpoch_  = 1000      # 1000
_MRnnPerEpoch_  = 1         # 1
_MRnnJudEpoch_  = 10        # 10
_PRnnMaxEpoch_  = 1000      # 1000
_PRnnPerEpoch_  = 1         # 1
_PRnnJudEpoch_  = 10        # 10
_TRnnMaxEpoch_  = 1         # 1000
_TRnnPerEpoch_  = 1         # 1
_TRnnJudEpoch_  = 10        # 10
# hyper-parameters. (flow control)
_DEBUG_ = 0 #  0 : release
            #  1 : debug
_LOCK_  = 0 #  0 : unlocked - create random split sets.
            #  1 : locked   - use the saved split sets.
_MODEL_ = 1 #  0 : unlocked - train a new model.
            #  1 : locked   - load the saved model.
_DTYP_  = 1 #  0 : maintain both diff code and context code.
            #  1 : only maintain diff code.
_CTYP_  = 1 #  0 : maintain both the code and comments.
            #  1 : only maintain code and delete comments.
_NIND_ =  1 # -1 : not abstract tokens. (and will disable _NLIT_)
            #  0 : abstract identifiers with VAR/FUNC.
            #  1 : abstract identifiers with VARn/FUNCn.
_NLIT_  = 1 #  0 : abstract literals with LITERAL.
            #  1 : abstract literals with LITERAL/n.
_TWIN_  = 1 #  0 : only twin neural network.
            #  1 : twins + msg neural network.
# global variable.
start_time = time.time() #mark start time

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def RunTime():
    pTime = ' [TIME: ' + str(round((time.time() - start_time), 2)) + ' sec]'
    return pTime

def ReadData():
    '''
    Read data from the files.
    :return: data - a set of commit message, diff code, and labels.
    [[['', ...], [['', ...], ['', ...], ...], 0/1], ...]
    '''

    def ReadCommitMsg(filename):
        '''
        Read commit message from a file.
        :param filename: file name (string).
        :return: commitMsg - commit message.
        ['line', 'line', ...]
        '''

        fp = open(filename, encoding='utf-8', errors='ignore')  # get file point.
        lines = fp.readlines()  # read all lines.
        #numLines = len(lines)   # get the line number.
        #print(lines)

        # initialize commit message.
        commitMsg = []
        # get the wide range of commit message.
        for line in lines:
            if line.startswith('diff --git'):
                break
            else:
                commitMsg.append(line)
        #print(commitMsg)
        # process the head of commit message.
        while (1):
            headMsg = commitMsg[0]
            if (headMsg.startswith('From') or headMsg.startswith('Date:') or headMsg.startswith('Subject:')
                    or headMsg.startswith('commit') or headMsg.startswith('Author:')):
                commitMsg.pop(0)
            else:
                break
        #print(commitMsg)
        # process the tail of commit message.
        dashLines = [i for i in range(len(commitMsg))
                     if commitMsg[i].startswith('---')]  # finds all lines start with ---.
        if (len(dashLines)):
            lnum = dashLines[-1]  # last line number of ---
            marks = [1 if (' file changed, ' in commitMsg[i] or ' files changed, ' in commitMsg[i]) else 0
                     for i in range(lnum, len(commitMsg))]
            if (sum(marks)):
                for i in reversed(range(lnum, len(commitMsg))):
                    commitMsg.pop(i)
        #print(commitMsg)

        #msgShow = ''
        #for i in range(len(commitMsg)):
        #    msgShow += commitMsg[i]
        #print(msgShow)

        return commitMsg

    def ReadDiffLines(filename):
        '''
        Read diff code from a file.
        :param filename:  file name (string).
        :return: diffLines - diff code.
        [['line', ...], ['line', ...], ...]
        '''

        fp = open(filename, encoding='utf-8', errors='ignore')  # get file point.
        lines = fp.readlines()  # read all lines.
        numLines = len(lines)  # get the line number.
        # print(lines)

        atLines = [i for i in range(numLines) if lines[i].startswith('@@ ')]  # find all lines start with @@.
        atLines.append(numLines)
        # print(atLines)

        diffLines = []
        for nh in range(len(atLines) - 1):  # find all hunks.
            # print(atLines[nh], atLines[nh + 1])
            hunk = []
            for nl in range(atLines[nh] + 1, atLines[nh + 1]):
                # print(lines[nl], end='')
                if lines[nl].startswith('diff --git '):
                    break
                else:
                    hunk.append(lines[nl])
            diffLines.append(hunk)
            # print(hunk)
        # print(diffLines)
        # print(len(diffLines))

        # process the last hunk.
        lastHunk = diffLines[-1]
        numLastHunk = len(lastHunk)
        dashLines = [i for i in range(numLastHunk) if lastHunk[i].startswith('--')]
        if (len(dashLines)):
            lnum = dashLines[-1]
            for i in reversed(range(lnum, numLastHunk)):
                lastHunk.pop(i)
        # print(diffLines)
        # print(len(diffLines))

        return diffLines

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + 'testfilelist.txt', 'w')

    # initialize filelist.
    filelist = []
    for root, ds, fs in os.walk(testPath):
        for file in fs:
            if '.DS_Store' in file: continue
            filename = os.path.join(root, file).replace('\\', '/')
            filelist.append(filename)
    filelist = natsorted(filelist)

    # initialize data.
    data = []
    for filename in filelist:
        # print(filename)
        fp.write(filename + '\n')
        commitMsg = ReadCommitMsg(filename)
        diffLines = ReadDiffLines(filename)
        data.append([commitMsg, diffLines, 0])
    fp.close()

    # save dataLoaded.
    np.save(tempPath + '/testdata.npy', data, allow_pickle=True)
    np.save(tempPath + '/testfilelist.npy', filelist, allow_pickle=True)
    print('[INFO] <ReadData> Save ' + str(len(data)) + ' raw data to ' + tempPath + '/testdata.npy.' + RunTime())

    return data, filelist

def GetDiffProps(data):
    '''
    Get the properties of the code in diff files.
    :param data: [[[line, , ], [[line, , ], [line, , ], ...], 0/1], ...]
    :return: props - [[[tokens], [nums], [nums], 0/1], ...]
    '''

    def RemoveSign(line):
        '''
        Remove the sign (+/-) in the first character.
        :param line: a code line.
        :return: process line.
        '''

        return ' ' + line[1:] if (line[0] == '+') or (line[0] == '-') else line

    def GetClangTokens(line):
        '''
        Get the tokens of a line with the Clang tool.
        :param line: a code line.
        :return: tokens - ['tk', 'tk', ...] ('tk': string)
                 tokenTypes - [tkt, tkt, ...] (tkt: 1, 2, 3, 4, 5)
                 diffTypes - [dft, dft, ...] (dft: -1, 0, 1)
        '''

        # remove non-ascii
        line = line.encode("ascii", "ignore").decode()

        # defination.
        tokenClass = [clang.cindex.TokenKind.KEYWORD,      # 1
                      clang.cindex.TokenKind.IDENTIFIER,   # 2
                      clang.cindex.TokenKind.LITERAL,      # 3
                      clang.cindex.TokenKind.PUNCTUATION,  # 4
                      clang.cindex.TokenKind.COMMENT]      # 5
        tokenDict = {cls: index + 1 for index, cls in enumerate(tokenClass)}
        #print(tokenDict)

        # initialize.
        tokens = []
        tokenTypes = []
        diffTypes = []

        # clang sparser.
        idx = clang.cindex.Index.create()
        tu = idx.parse('tmp.cpp', args=['-std=c++11'], unsaved_files=[('tmp.cpp', RemoveSign(line))], options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            #print(t.kind, t.spelling, t.location)
            tokens.append(t.spelling)
            tokenTypes.append(tokenDict[t.kind])
            diffTypes.append(1 if (line[0] == '+') else -1 if (line[0] == '-') else 0)
        #print(tokens)
        #print(tokenTypes)
        #print(diffTypes)

        return tokens, tokenTypes, diffTypes

    def GetWordTokens(line):
        '''
        Get the word tokens from a code line.
        :param line: a code line.
        :return: tokens - ['tk', 'tk', ...] ('tk': string)
        '''

        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(RemoveSign(line))
        return tokens

    def GetString(lines):
        '''
        Get the strings from the diff code
        :param lines: diff code.
        :return: lineStr - All the diff lines.
                 lineStrB - The before-version code lines.
                 lineStrA - The after-version code lines.
        '''

        lineStr = ''
        lineStrB = ''
        lineStrA = ''
        for hunk in lines:
            for line in hunk:
                # all lines.
                lineStr += RemoveSign(line)
                # all Before lines.
                lineStrB += RemoveSign(line) if line[0] != '+' else ''
                # all After lines.
                lineStrA += RemoveSign(line) if line[0] != '-' else ''

        return lineStr, lineStrB, lineStrA

    def GetDiffTokens(lines):
        '''
        Get the tokens for the diff lines.
        :param lines: the diff code.
        :return: tokens - tokens ['tk', 'tk', ...] ('tk': string)
                 tokenTypes - token types [tkt, tkt, ...] (tkt: 1, 2, 3, 4, 5)
                 diffTypes - diff types [dft, dft, ...] (dft: -1, 0, 1)
        '''

        # initialize.
        tokens = []
        tokenTypes = []
        diffTypes = []

        # for each line of lines.
        for hunk in lines:
            for line in hunk:
                #print(line, end='')
                tk, tkT, dfT = GetClangTokens(line)
                tokens.extend(tk)
                tokenTypes.extend(tkT)
                diffTypes.extend(dfT)
                #print('-----------------------------------------------------------------------')
        #print(tokens)
        #print(tokenTypes)
        #print(diffTypes)

        return tokens, tokenTypes, diffTypes

    #lines = data[0][1]
    #print(lines)
    #hunk = data[0][1][0]
    #print(hunk)
    #line = data[0][1][0][0]
    #print(line)

    # for each sample data[n].
    numData = len(data)
    props = []
    for n in range(numData):
        # get the lines of the diff file.
        diffLines = data[n][1]
        # properties.
        tk, tkT, dfT = GetDiffTokens(diffLines)
        label = data[n][2]
        prop = [tk, tkT, dfT, label]
        #print(prop)
        props.append(prop)
        print('[INFO] <GetDiffProps> Processing Tokens ' + str(n+1) + '/' + str(numData) + '...' + RunTime())

    # save props.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)

    np.save(tempPath + '/testprops.npy', props, allow_pickle=True)
    print('[INFO] <GetDiffProps> Save ' + str(len(props)) + ' diff property data to ' + tempPath + '/testprops.npy.' + RunTime())

    return props

def ProcessTokens(props, dType=1, cType=1):
    '''
    only maintain the diff parts of the code.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param dType: 0 - maintain both diff code and context code.
                  1 - only maintain diff code.
    :param cType: 0 - maintain both the code and comments.
                  1 - only maintain code and delete comments.
    :return: props - the normalized features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    '''

    # process diff code.
    if (1 == dType):
        propsNew = []
        for item in props:
            # the number of tokens.
            numTokens = len(item[1])
            # item[0]: tokens, item[1]: tokenTypes, item[2]: diffTypes, item[3]: label.
            tokens = [item[0][n] for n in range(numTokens) if (item[2][n])]
            tokenTypes = [item[1][n] for n in range(numTokens) if (item[2][n])]
            diffTypes = [item[2][n] for n in range(numTokens) if (item[2][n])]
            label = item[3]
            # reconstruct sample.
            sample = [tokens, tokenTypes, diffTypes, label]
            propsNew.append(sample)
        props = propsNew
        print('[INFO] <ProcessTokens> Only maintain the diff parts of the code.' + RunTime())

    # process comments.
    if (1 == cType):
        propsNew = []
        for item in props:
            # the number of tokens.
            numTokens = len(item[1])
            # item[0]: tokens, item[1]: tokenTypes, item[2]: diffTypes, item[3]: label.
            tokens = [item[0][n] for n in range(numTokens) if (item[1][n] < 5)]
            tokenTypes = [item[1][n] for n in range(numTokens) if (item[1][n] < 5)]
            diffTypes = [item[2][n] for n in range(numTokens) if (item[1][n] < 5)]
            label = item[3]
            # reconstruct sample.
            sample = [tokens, tokenTypes, diffTypes, label]
            propsNew.append(sample)
        props = propsNew
        print('[INFO] <ProcessTokens> Delete the comment parts of the diff code.' + RunTime())

    #print(props[0])

    return props

def AbstractTokens(props, iType=1, lType=1):
    '''
    abstract the tokens of identifiers, literals, and comments.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param iType:   -1 - not abstract tokens.
                     0 - only abstract variable type and function type. VAR / FUNC
                     1 - abstract the identical variable names and function names.  VAR0, VAR1, ... / FUNC0, FUNC1, ...
    :param lType:   -1 - not abstract tokens.
                     0 - abstract literals with LITERAL.
                     1 - abstract literals with LITERAL/n.
    :return: props - the abstracted features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    '''

    if (iType not in [0, 1]) or (lType not in [0, 1]):
        print('[INFO] <AbstractTokens> Not abstract the tokens of identifiers, literals, and comments.' + RunTime())
        return props

    for item in props:
        # get tokens and token types.
        tokens = item[0]
        tokenTypes = item[1]
        numTokens = len(tokenTypes)
        #print(tokens)
        #print(tokenTypes)
        #print(numTokens)

        # abstract literals and comments, and separate identifiers into variables and functions.
        markVar = list(np.zeros(numTokens, dtype=int))
        markFuc = list(np.zeros(numTokens, dtype=int))
        for n in range(numTokens):
            # 2: IDENTIFIER, 3: LITERAL, 5: COMMENT
            if 5 == tokenTypes[n]:
                tokens[n] = 'COMMENT'
            elif 3 == tokenTypes[n]:
                if (0 == lType):
                    tokens[n] = 'LITERAL'
                elif (1 == lType):
                    if (not tokens[n].isdigit()):
                        tokens[n] = 'LITERAL'
            elif 2 == tokenTypes[n]:
                # separate variable name and function name.
                if (n < numTokens-1):
                    if (tokens[n+1] == '('):
                        markFuc[n] = 1
                    else:
                        markVar[n] = 1
                else:
                    markVar[n] = 1
        #print(tokens)
        #print(markVar)
        #print(markFuc)

        # abstract variables and functions.
        if (0 == iType):
            for n in range(numTokens):
                if 1 == markVar[n]:
                    tokens[n] = 'VAR'
                elif 1 == markFuc[n]:
                    tokens[n] = 'FUNC'
        elif (1 == iType):
            # get variable dictionary.
            varList = [tokens[idx] for idx, mark in enumerate(markVar) if mark == 1]
            varVoc  = {}.fromkeys(varList)
            varVoc  = list(varVoc.keys())
            varDict = {tk: 'VAR' + str(idx) for idx, tk in enumerate(varVoc)}
            # get function dictionary.
            fucList = [tokens[idx] for idx, mark in enumerate(markFuc) if mark == 1]
            fucVoc  = {}.fromkeys(fucList)
            fucVoc  = list(fucVoc.keys())
            fucDict = {tk: 'FUNC' + str(idx) for idx, tk in enumerate(fucVoc)}
            #print(varDict)
            #print(fucDict)
            for n in range(numTokens):
                if 1 == markVar[n]:
                    tokens[n] = varDict[tokens[n]]
                elif 1 == markFuc[n]:
                    tokens[n] = fucDict[tokens[n]]
    #print(tokens)
    print('[INFO] <AbstractTokens> Abstract the tokens of identifiers with iType ' + str(iType), end='')
    print(' (VAR/FUNC).' + RunTime()) if (0 == iType) else print(' (VARn/FUNCn).' + RunTime())
    print('[INFO] <AbstractTokens> Abstract the tokens of literals, and comments with iType ' + str(lType), end='')
    print(' (LITERAL/COMMENT).' + RunTime()) if (0 == lType) else print(' (LITERAL/n/COMMENT).' + RunTime())

    return props

def GetDiffEmbed(tokenDict, embedSize):
    '''
    Get the pre-trained weights for embedding layer from the dictionary of diff vocabulary.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    '''

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize,))
    print('[INFO] <GetDiffEmbed> Create pre-trained embedding weights with ' + str(len(preWeights)) + ' * ' + str(len(preWeights[0])) + ' matrix.' + RunTime())

    return preWeights

def DivideBeforeAfter(diffProps):

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    # fp = open(tempPath + 'testtwinlen.csv', 'w')

    twinProps = []
    maxLen = 0
    # for each sample in diffProps.
    for item in diffProps:
        # get the tk, tkT, dfT, lb.
        tokens = item[0]
        tokenTypes = item[1]
        diffTypes = item[2]
        label = item[3]
        numTokens = len(diffTypes)
        # reconstruct tkB, tkTB, tkA, tkTA.
        tokensB = [tokens[i] for i in range(numTokens) if (diffTypes[i] <= 0)]
        tokenTypesB = [tokenTypes[i] for i in range(numTokens) if (diffTypes[i] <= 0)]
        tokensA = [tokens[i] for i in range(numTokens) if (diffTypes[i] >= 0)]
        tokenTypesA = [tokenTypes[i] for i in range(numTokens) if (diffTypes[i] >= 0)]
        # reconstruct new sample.
        sample = [tokensB, tokenTypesB, tokensA, tokenTypesA, label]
        twinProps.append(sample)
        # get max length.
        maxLenAB = max(len(tokenTypesB), len(tokenTypesA))
        maxLen = maxLenAB if (maxLen < maxLenAB) else maxLen
    #     fp.write(str(len(tokenTypesB)) + '\n')
    #     fp.write(str(len(tokenTypesA)) + '\n')
    # fp.close()

    #print(twinProps[0])
    #print(maxLen)

    # print.
    print('[INFO] <DivideBeforeAfter> Divide diff code into BEFORE-version and AFTER-version code.' + RunTime())
    print('[INFO] <DivideBeforeAfter> The max length in BEFORE/AFTER-version code is ' + str(maxLen) + ' tokens. (hyperparameter: _TwinMaxLen_ = ' + str(_TwinMaxLen_) + ')' + RunTime())

    return twinProps, maxLen

def GetTwinMapping(props, maxLen, tokenDict):
    '''
    Map the feature data into indexed data.
    :param props: the features of diff code.
    [[[tokens], [nums], [tokens], [nums], 0/1], ...]
    :param maxLen: the max length of a twin code.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[[n, {0~5}, n, {0~5}], ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    '''

    def PadList(dList, pad, length):
        '''
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        '''

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in props:
        # initialize sample.
        sample = []

        # process tokensB.
        tokens = item[0]
        tokens = PadList(tokens, '<pad>', maxLen)
        tokens2index = []
        for tk in tokens:
            if (tk in tokenDict.keys()):
                tokens2index.append(tokenDict[tk])
            else:
                tokens2index.append(0)
        sample.append(tokens2index)
        # process tokenTypesB.
        tokenTypes = item[1]
        tokenTypes = PadList(tokenTypes, 0, maxLen)
        sample.append(tokenTypes)
        # process tokensA.
        tokens = item[2]
        tokens = PadList(tokens, '<pad>', maxLen)
        tokens2index = []
        for tk in tokens:
            if (tk in tokenDict.keys()):
                tokens2index.append(tokenDict[tk])
            else:
                tokens2index.append(0)
        sample.append(tokens2index)
        # process tokenTypesA.
        tokenTypes = item[3]
        tokenTypes = PadList(tokenTypes, 0, maxLen)
        sample.append(tokenTypes)

        # process sample.
        sample = np.array(sample).T
        data.append(sample)
        # process label.
        label = item[4]
        labels.append([label])

    if _DEBUG_:
        print('[DEBUG] data:')
        print(data[0:3])
        print('[DEBUG] labels:')
        print(labels[0:3])

    # print.
    print('[INFO] <GetTwinMapping> Create ' + str(len(data)) + ' feature data with ' + str(len(data[0])) + ' * ' + str(len(data[0][0])) + ' matrix.' + RunTime())
    print('[INFO] <GetTwinMapping> Create ' + str(len(labels)) + ' labels with 1 * 1 matrix.' + RunTime())

    # save files.
    # if (not os.path.exists(tempPath + '/tdata_' + str(maxLen) + '.npy')) \
    #         | (not os.path.exists(tempPath + '/tlabels_' + str(maxLen) + '.npy')):
    #     np.save(tempPath + '/tdata_' + str(maxLen) + '.npy', data, allow_pickle=True)
    #     print('[INFO] <GetTwinMapping> Save the mapped numpy data to ' + tempPath + '/tdata_' + str(maxLen) + '.npy.')
    #     np.save(tempPath + '/tlabels_' + str(maxLen) + '.npy', labels, allow_pickle=True)
    #     print('[INFO] <GetTwinMapping> Save the mapped numpy labels to ' + tempPath + '/tlabels_' + str(maxLen) + '.npy.')

    return np.array(data), np.array(labels)

def UpdateTwinTokenTypes(data):
    '''
    Update the token type in the feature data into one-hot vector.
    :param data: feature data. [[[n, {0~5}, n, {0~5},], ...], ...]
    :return: np.array(newData). [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, n, 0/1, 0/1, 0/1, 0/1, 0/1], ...], ...]
    '''

    newData = []
    # for each sample.
    for item in data:
        # get the transpose of props.
        itemT = item.T
        # initialize new sample.
        newItem = []
        newItem.append(itemT[0])
        newItem.extend(np.zeros((5, len(item)), dtype=int))
        newItem.append(itemT[2])
        newItem.extend(np.zeros((5, len(item)), dtype=int))
        # assign the new sample.
        for i in range(len(item)):
            tokenType = itemT[1][i]
            if (tokenType):
                newItem[tokenType][i] = 1
            tokenType = itemT[3][i]
            if (tokenType):
                newItem[tokenType+6][i] = 1
        # get the transpose of new sample.
        newItem = np.array(newItem).T
        # append new sample.
        newData.append(newItem)

    if _DEBUG_:
        print('[DEBUG] newData:')
        print(newData[0:3])

    # print.
    print('[INFO] <UpdateTwinTokenTypes> Update ' + str(len(newData)) + ' feature data with ' + str(len(newData[0])) + ' * ' + str(len(newData[0][0])) + ' matrix.' + RunTime())

    # save files.
    # if (not os.path.exists(tempPath + '/newtdata_' + str(len(newData[0])) + '.npy')):
    #     np.save(tempPath + '/newtdata_' + str(len(newData[0])) + '.npy', newData, allow_pickle=True)
    #     print('[INFO] <UpdateTwinTokenTypes> Save the mapped numpy data to ' + tempPath + '/newtdata_' + str(len(newData[0])) + '.npy.')

    # change marco.
    global _TwinExtraDim_
    _TwinExtraDim_ = 5

    return np.array(newData)

def GetCommitMsgs(data):
    '''
    Get the commit messages in diff files.
    :param data: [[[line, , ], [[line, , ], [line, , ], ...], 0/1], ...]
    :return: msgs - [[[tokens], 0/1], ...]
    '''

    def GetMsgTokens(lines):
        '''
        Get the tokens from a commit message.
        :param lines: commit message. [line, , ]
        :return: tokensStem ['tk', , ]
        '''

        # concatenate lines.
        # get the string of commit message.
        msg = ''
        for line in lines:
            msg += line[:-1] + ' '
        #print(msg)

        # pre-process.
        # remove url.
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        msg = re.sub(pattern, ' ', msg)
        # remove independent numbers.
        pattern = r' \d+ '
        msg = re.sub(pattern, ' ', msg)
        # lower case capitalized words.
        pattern = r'([A-Z][a-z]+)'
        def LowerFunc(matched):
            return matched.group(1).lower()
        msg = re.sub(pattern, LowerFunc, msg)
        # remove footnote.
        patterns = ['signed-off-by:', 'reported-by:', 'reviewed-by:', 'acked-by:', 'found-by:', 'tested-by:', 'cc:']
        for pattern in patterns:
            index = msg.find(pattern)
            if (index > 0):
                msg = msg[:index]
        #print(msg)

        # clearance.
        # get the tokens.
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(msg)
        # clear tokens that don't contain any english letter.
        for i in reversed(range(len(tokens))):
            if not (re.search('[a-z]', tokens[i])):
                tokens.pop(i)
        # clear tokens that are stopwords.
        for i in reversed(range(len(tokens))):
            if (tokens[i] in stopwords.words('english')):
                tokens.pop(i)
        pattern = re.compile("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
        for i in reversed(range(len(tokens))):
            if (pattern.findall(tokens[i])):
                tokens.pop(i)
        #print(tokens)

        # process tokens with stemming.
        porter = PorterStemmer()
        tokensStem = []
        for item in tokens:
            tokensStem.append(porter.stem(item))
        #print(tokensStem)

        return tokensStem

    # for each sample data[n].
    numData = len(data)
    msgs = []
    for n in range(numData):
        # get the lines of the commit message.
        commitMsg = data[n][0]
        mtk = GetMsgTokens(commitMsg)
        # get the label.
        label = data[n][2]
        #print([mtk, label])
        # append the message tokens.
        msgs.append([mtk, label])
        print('[INFO] <GetCommitMsg> Processing Commit ' + str(n+1) + '/' + str(numData) + '...' + RunTime())

    # save commit messages.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    np.save(tempPath + '/testmsgs.npy', msgs, allow_pickle=True)
    print('[INFO] <GetCommitMsg> Save ' + str(len(msgs)) + ' commit messages to ' + tempPath + '/testmsgs.npy.' + RunTime())

    return msgs

def GetMsgEmbed(tokenDict, embedSize):
    '''
    Get the pre-trained weights for embedding layer from the dictionary of msg vocabulary.
    :param tokenDict: the dictionary of msg vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    '''

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize,))
    print('[INFO] <GetMsgEmbed> Create pre-trained embedding weights with ' + str(len(preWeights)) + ' * ' + str(len(preWeights[0])) + ' matrix.' + RunTime())

    return preWeights

def GetMsgMapping(msgs, maxLen, tokenDict):
    '''
    Map the feature data into indexed data.
    :param props: the features of commit messages.
    [[[tokens], 0/1], ...]
    :param maxLen: the max length of the commit message.
    :param tokenDict: the dictionary of commit message vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[n, ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    '''

    def PadList(dList, pad, length):
        '''
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        '''

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in msgs:
        # process tokens.
        tokens = item[0]
        tokens = PadList(tokens, '<pad>', maxLen)
        # convert tokens into numbers.
        tokens2index = []
        for tk in tokens:
            if (tk in tokenDict.keys()):
                tokens2index.append(tokenDict[tk])
            else:
                tokens2index.append(0)
        data.append(tokens2index)
        # process label.
        label = item[1]
        labels.append([label])

    if _DEBUG_:
        print('[DEBUG] data:')
        print(data[0:3])
        print('[DEBUG] labels:')
        print(labels[0:3])

    # print.
    print('[INFO] <GetMsgMapping> Create ' + str(len(data)) + ' feature data with 1 * ' + str(len(data[0])) + ' vector.' + RunTime())
    print('[INFO] <GetMsgMapping> Create ' + str(len(labels)) + ' labels with 1 * 1 matrix.' + RunTime())

    # # save files.
    # if (not os.path.exists(tempPath + '/mdata_' + str(maxLen) + '.npy')) \
    #         | (not os.path.exists(tempPath + '/mlabels_' + str(maxLen) + '.npy')):
    #     np.save(tempPath + '/mdata_' + str(maxLen) + '.npy', data, allow_pickle=True)
    #     print('[INFO] <GetMsgMapping> Save the mapped numpy data to ' + tempPath + '/mdata_' + str(maxLen) + '.npy.')
    #     np.save(tempPath + '/mlabels_' + str(maxLen) + '.npy', labels, allow_pickle=True)
    #     print('[INFO] <GetMsgMapping> Save the mapped numpy labels to ' + tempPath + '/mlabels_' + str(maxLen) + '.npy.')

    return np.array(data), np.array(labels)

def CombineTwinMsgs(props, msgs, plabels, mlabels):
    '''
    Combine the twin props with the commit messages.
    :param props: twin data. [[[n, {0~5}, n, {0~5}], ...], ...] or [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, n, 0/1, 0/1, 0/1, 0/1, 0/1], ...], ...]
    :param msgs: message data. [[n, ...], ...]
    :param plabels: twin labels. [[0/1], ...]
    :param mlabels: message labels. [[0/1], ...]
    :return: np.array(data) - combined data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, n, 0/1, 0/1, 0/1, 0/1, 0/1, n], ...], ...]
             np.array(plabels) - combined labels. [[0/1], ...]
    '''

    # check the lengths.
    if (len(plabels) != len(mlabels)):
        print('[ERROR] <CombineTwinMsgs> the data lengths are mismatch.')
        return [[]], [[]]

    # check the labels.
    cntMatch = 0
    for n in range(len(plabels)):
        if (plabels[n][0] == mlabels[n][0]):
            cntMatch += 1
    if (cntMatch != len(plabels)):
        print('[ERROR] <CombineTwinMsgs> the labels are mismatch. ' + str(cntMatch) + '/' + str(len(plabels)) + '.')
        return [[]], [[]]

    #print(props[1], len(props[1]))
    #print(msgs[1], len(msgs[1]))

    data = []
    for n in range(len(plabels)):
        # get the twin prop and message.
        prop = props[n]
        msg = msgs[n]
        # pad data.
        if (_TwinMaxLen_ >= _MsgMaxLen_):
            msg = np.pad(msg, (0, _TwinMaxLen_ - _MsgMaxLen_), 'constant')
        else:
            prop = np.pad(prop, ((0, _MsgMaxLen_ - _TwinMaxLen_), (0, 0)), 'constant')
        #print(msg, len(msg))
        #print(prop, len(prop))
        # reconstruct sample.
        sample = np.vstack((prop.T, msg))
        # append the sample to data.
        data.append(sample.T)

    if _DEBUG_:
        print(np.array(data[0:3]))

    print('[INFO] <CombineTwinMsgs> Combine the twin props with the commit messages.' + RunTime())

    return np.array(data), np.array(plabels)

class TwinRNN(nn.Module):
    '''
    TwinRNN : convert a patch data into a predicted label.
    '''

    def __init__(self, preWTwin, preWMsg, hidSizTwin=32, hidSizMsg=32, hidLayTwin=1, hidLayMsg=1):
        '''
        define each layer in the network model.
        :param preWTwin: tensor pre-trained weights for embedding layer for twin.
        :param preWMsg: tensor pre-trained weights for embedding layer for msg.
        :param hidSizTwin: node number in the hidden layer for twin.
        :param hidSizMsg: node number in the hidden layer for msg.
        :param hidLayTwin: number of hidden layer for twin.
        :param hidLayMsg: number of hidden layer for msg.
        '''

        super(TwinRNN, self).__init__()
        # parameters.
        class_num = 2
    # twin.
        vSizTwin, emDimTwin = preWTwin.size()
        # Embedding Layer for twin.
        self.embedTwin = nn.Embedding(num_embeddings=vSizTwin, embedding_dim=emDimTwin)
        self.embedTwin.load_state_dict({'weight': preWTwin})
        self.embedTwin.weight.requires_grad = True
        # LSTM Layer for twin.
        if _DEBUG_: print(_TwinExtraDim_)
        self.lstmTwin = nn.LSTM(input_size=emDimTwin+_TwinExtraDim_, hidden_size=hidSizTwin, num_layers=hidLayTwin, bidirectional=True)
    # msg.
        vSizMsg, emDimMsg = preWMsg.size()
        # Embedding Layer for msg.
        self.embedMsg = nn.Embedding(num_embeddings=vSizMsg, embedding_dim=emDimMsg)
        self.embedMsg.load_state_dict({'weight': preWMsg})
        self.embedMsg.weight.requires_grad = True
        # LSTM Layer for msg.
        self.lstmMsg = nn.LSTM(input_size=emDimMsg, hidden_size=hidSizMsg, num_layers=hidLayMsg, bidirectional=True)
    # common.
        # Fully-Connected Layer.
        self.fc1 = nn.Linear(hidSizTwin * hidLayTwin * 4, hidSizTwin * hidLayTwin * 2)
        self.fc2 = nn.Linear(hidSizTwin * hidLayTwin * 2, class_num)
        self.fc3 = nn.Linear((hidSizTwin * hidLayTwin + hidSizMsg * hidLayMsg) * 2, hidSizTwin * hidLayTwin + hidSizMsg * hidLayMsg)
        self.fc4 = nn.Linear(hidSizTwin * hidLayTwin + hidSizMsg * hidLayMsg, class_num)
        # Softmax non-linearity.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * twin_length * feature_dim.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        '''

    # twin 1.
        xTwin = x[:, :_TwinMaxLen_, :6]
        # xTwin         batch_size * twin_length * feature_dim
        #print(xTwin.size())
        embedsTwin = self.embedTwin(xTwin[:, :, 0])
        # embedsTwin    batch_size * twin_length * embed_dim_twin
        features = xTwin[:, :, 1:]
        # features      batch_size * twin_length * _TwinExtraDim_
        inputsTwin = torch.cat((embedsTwin.float(), features.float()), 2)
        #print(inputsTwin.size())
        # inputsTwin    batch_size * twin_length * (embed_dim_twin + _TwinExtraDim_)
        inputsTwin = inputsTwin.permute(1, 0, 2)
        # inputsTwin    twin_length * batch_size * (embed_dim_twin + _TwinExtraDim_)
        lstm_out, (h_n, c_n) = self.lstmTwin(inputsTwin)
        # lstm_out      twin_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapTwin1 = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapTwin1   batch_size * (hidden_size * num_layers * direction_num)
        #print(featMapTwin1)
    # twin 2.
        xTwin = x[:, :_TwinMaxLen_, 6:-1]
        # xTwin         batch_size * twin_length * feature_dim
        #print(xTwin.size())
        embedsTwin = self.embedTwin(xTwin[:, :, 0])
        # embedsTwin    batch_size * twin_length * embed_dim_twin
        features = xTwin[:, :, 1:]
        # features      batch_size * twin_length * _TwinExtraDim_
        inputsTwin = torch.cat((embedsTwin.float(), features.float()), 2)
        #print(inputsTwin.size())
        # inputsTwin    batch_size * twin_length * (embed_dim_twin + _TwinExtraDim_)
        inputsTwin = inputsTwin.permute(1, 0, 2)
        # inputsTwin    twin_length * batch_size * (embed_dim_twin + _TwinExtraDim_)
        lstm_out, (h_n, c_n) = self.lstmTwin(inputsTwin)
        # lstm_out      twin_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapTwin2 = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapTwin2   batch_size * (hidden_size * num_layers * direction_num)
        #print(featMapTwin2)
    # msg.
        xMsg = x[:, :_MsgMaxLen_, -1]
        # xMsg          batch_size * msg_length * 1
        # print(xMsg.size())
        embedsMsg = self.embedMsg(xMsg)
        # embedsMsg     batch_size * msg_length * embed_dim_msg
        inputsMsg = embedsMsg.permute(1, 0, 2)
        # inputsMsg     msg_length * batch_size * (embed_dim_msg)
        lstm_out, (h_n, c_n) = self.lstmMsg(inputsMsg)
        # lstm_out      msg_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        featMapMsg = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # featMapMsg    batch_size * (hidden_size * num_layers * direction_num)
        #print(featMapMsg.size())
    # common.
        # combine twins.
        featMap = torch.cat((featMapTwin1, featMapTwin2), dim=1)
        # fc layers.
        featMap = self.fc1(featMap)
        if (0 == _TWIN_): # (only twins).
            final_out = self.fc2(featMap)
        elif (1 == _TWIN_): # (twins + msg).
            # combine twins + msg.
            featMap = torch.cat((featMap, featMapMsg), dim=1)
            # fc 2 layers.
            featMap = self.fc3(featMap)
            final_out = self.fc4(featMap)
        #print(final_out.size())
        return self.softmax(final_out)      # batch_size * class_num

def TwinRNNTest(model, dTest, lTest, batchsize=64):
    '''
    Test the TwinRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    '''

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long()
    yTest = torch.from_numpy(lTest).long()
    if (torch.cuda.is_available()):
        xTest = xTest.cuda()
        yTest = yTest.cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    print('[INFO] <TwinRNNTest> Finish the prediction of test data.' + RunTime())

    return predictions, accuracy

def demoTwin():
    # load data.
    dataLoaded, filelist = ReadData()

    # get the diff file properties.
    # if (not os.path.exists(tempPath + '/testprops.npy')):
    diffProps = GetDiffProps(dataLoaded)
    # else:
    #     diffProps = np.load(tempPath + '/testprops.npy', allow_pickle=True)
    #     print('[INFO] <GetDiffProps> Load ' + str(len(diffProps)) + ' diff property data from ' + tempPath + '/testprops.npy.' + RunTime())

    # maintain both the context and diff parts. Delete comments.
    diffProps = ProcessTokens(diffProps, dType=0, cType=_CTYP_)
    # normalize the tokens of identifiers (VARn/FUNCn), literals (LITERAL/n), and comments (none).
    diffProps = AbstractTokens(diffProps, iType=_NIND_, lType=_NLIT_)
    # get the diff token dictionary.
    with open(modlPath + '/tokenDict.pickle', 'rb') as handle:
        diffDict = pickle.load(handle)
    # get pre-trained weights for embedding layer.
    twinPreWeights = GetDiffEmbed(diffDict, _TwinEmbedDim_)
    # divide diff code into before/after-version code.
    twinProps, twinMaxLen = DivideBeforeAfter(diffProps)
    # get the max twin length.
    twinMaxLen = _TwinMaxLen_
    # get the mapping for feature data and labels.
    twinData, twinLabels = GetTwinMapping(twinProps, twinMaxLen, diffDict)
    # change the tokentypes into one-hot vector.
    twinData = UpdateTwinTokenTypes(twinData)

    # get the commit messages from data.
    # if (not os.path.exists(tempPath + '/testmsgs.npy')):
    commitMsgs = GetCommitMsgs(dataLoaded)
    # else:
    #     commitMsgs = np.load(tempPath + '/testmsgs.npy', allow_pickle=True)
    #     print('[INFO] <GetCommitMsg> Load ' + str(len(commitMsgs)) + ' commit messages from ' + tempPath + '/testmsgs.npy.' + RunTime())
    # get the msg token dictionary.
    with open(modlPath + '/tokenDictMsg.pickle', 'rb') as handle:
        msgDict = pickle.load(handle)
    # get pre-trained weights for embedding layer.
    msgPreWeights = GetMsgEmbed(msgDict, _MsgEmbedDim_)
    # get the max msg length.
    msgMaxLen = _MsgMaxLen_
    # get the mapping for feature data and labels.
    msgData, msgLabels = GetMsgMapping(commitMsgs, msgMaxLen, msgDict)

    # combine the twin data with the message data.
    data, label = CombineTwinMsgs(twinData, msgData, twinLabels, msgLabels)

    # TwinRNN
    if (_MODEL_) & (os.path.exists(modlPath + '/model_TwinRNN.pth')):
        preWTwin = torch.from_numpy(twinPreWeights)
        preWMsg = torch.from_numpy(msgPreWeights)
        model = TwinRNN(preWTwin, preWMsg, hidSizTwin=_TRnnHidSiz_, hidSizMsg=_MRnnHidSiz_, hidLayTwin=_TRnnHidLay_, hidLayMsg=_MRnnHidLay_)
        if (torch.cuda.is_available()):
            model.load_state_dict(torch.load(modlPath + '/model_TwinRNN.pth'))
            print('[INFO] <demoTwin> Load model parameters with GPU-torch!' + RunTime())
        else:
            model.load_state_dict(torch.load(modlPath + '/model_TwinRNN.pth', map_location=torch.device('cpu')))
            print('[INFO] <demoTwin> Load model parameters with CPU-torch!' + RunTime())
    else:
        print('[ERROR] <demoTwin> Cannot find ' + modlPath + '/model_TwinRNN.pth' + RunTime())

    # TwinRNNTest
    predictions, accuracy = TwinRNNTest(model, data, label, batchsize=_TRnnBatchSz_)
    SaveResults(filelist, predictions)

    return

def SaveResults(filelist, predictions):
    '''
    :param filelist:
    :param predictions:
    :return:
    '''

    num = len(filelist)
    if not os.path.exists(resuPath):
        os.mkdir(resuPath)
    fp = open(resuPath + 'results.txt', 'w')

    for i in range(num):
        fp.write(filelist[i] + ',' + str(predictions[i][0]) + '\n')
    fp.close()

    print('[INFO] <SaveResults> Save the final results in ' + resuPath + 'results.txt' + RunTime())

    return

if __name__ == '__main__':
    # initialize the log file.
    logfile = 'PatchRNNdemo.log'
    if os.path.exists(os.path.join(logsPath, logfile)):
        os.remove(os.path.join(logsPath, logfile))
    elif not os.path.exists(logsPath):
        os.makedirs(logsPath)
    sys.stdout = Logger(os.path.join(logsPath, logfile))
    # demo
    demoTwin()