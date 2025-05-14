# -*- coding: utf-8 -*-

"""
Created on Tue March 5 2024, last updated on May 14 2025
Author: Michael Wittweiler based on work by Franziska Schropp and Marie Revellio

This work was carried out as part of the project "Zitieren als narrative Strategie. Eine digital-hermeneutische Untersuchung von Intertextualitätsphänomenen am Beispiel des Briefcorpus des Kirchenlehrers Hieronymus." under the supervision of Prof. Dr. Barbara Feichtinger and Dr. Marie Revellio, and was supported by the German Research Foundation (DFG, Forschungsgemeinschaft) [382880410].
"""

# Imports
import csv
import time
import re
from typing import List, Tuple, Any, Dict, Set, Union
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer 
import torch
import string
import itertools
from itertools import combinations
import spacy 
from spacy.language import Language
import pandas as pd
import openpyxl
import numpy
from numpy import dot
from numpy.linalg import norm
import argparse

# Adjustable global variables --- EXPERIMENTAL ---
min_number_of_shared_words = 2 # int, range(0,n); Used in the compare_text() function. 
min_number_complura = 4 # int, range(0,n); Used in the compare_text() function. Works as an additional layer to find potential citations that consist of longer collocations. A collocation of specified length or longer will be caught IGNORING stopwords and stored seperately from the other processes.
maximum_distance_between_shared_words = 3 # int, range(0,n); Used in the apply_distance_criterion() in compare_text(): In both passages the two shared words must be closer or as close to each other as the value of this variable. Stopwords are not counted towards this value. ATTENTION: For the calculation indices are subtracted, so 2 adjacent words have a distance of 1.  
similarity_threshold = 0.3 # float, range(0,1); context similarity filter: A cosine similarity of the word embeddings between the two shared words ABOVE this threshold is considered irrelevant and filtered out

# Models for HTRG and similarity filter
BASE_MODEL_NAME = "enelpol/evalatin2022-pos-open"
spacy_model = "la_core_web_lg"

### CALLGRAPH ###
"""
main()
  |
  |----> argparse()  
  |
  |----> assimilation(input_text_1)
  |         |
  |         '----> read_in_text(input_text)
  |         '----> assimilate(my_list)
  |                   |
  |                   '----> tokenizer(text)
  |                   '----> transform_token(token)
  |
  |----> assimilation(input_text_2)
  |         |
  |         '----> read_in_text(input_text)
  |         '----> assimilate(my_list)
  |                   |
  |                   '----> tokenizer(text)
  |                   '----> transform_token(token)
  |
  |----> phrasing_poetry(assimilated_list_1, output_phrasing_1)
  |   or
  |----> phrasing_prose(assimilated_list_2, output_phrasing_2)
  |         |
  |         '----> normalize_quotation_marks(text_list)
  |         '----> remove_whitespace_before_connectors(text_list)
  |         '----> strip_whitespaces(text_list)
  |         '----> remove_left_over_counts(text_list)
  |         '----> separate_text_at_ellipsis(text_list)
  |         '----> separate_text_at_punctuation_marks(text_list)
  |         '----> cleanup(text_list)
  |         '----> limited_bracket_contraction(text_list)
  |         '----> cleanup(text_list)
  |         '----> apply_condition_and_action(text_list, condition_func, action_func) (5 times with different condition functions)
  |         '----> cleanup(text_list)
  |         '----> merge_if_no_terminal_punctuation(text_list)
  |         '----> cleanup(text_list)
  |
  |----> compare_texts(phrasing_list_1, phrasing_list_2, stop_list)
  |         |
  |         '----> separate_text_meta(text_list_1, text_list_2)
  |         '----> normalize_orthographic_variants(pure_text_list)
  |         '----> normalize_typography(pure_text_list)
  |         '----> remove_unwanted_characters(pure_text_list)
  |         '----> tokenize_text(pure_text_list)
  |         '----> initialize_stopwords(filename)
  |         '----> text_matching(tokens_1, tokens_2, ...)
  |                   |
  |                   '----> find_four_adjacent(indices_a)
  |                   '----> find_four_adjacent(indices_b)
  |                   '----> highlight_shared_words(complura_sentence_a, complura_shared)
  |                   '----> highlight_shared_words(complura_sentence_b, complura_shared)
  |                   '----> highlight_shared_words(sentence_a, matched_items)
  |                   '----> highlight_shared_words(sentence_b, matched_items)
  |                   '----> write_to_excel(complura_matches_meta, output_complura)
  |         |
  |         |
  |         '----> apply_distance_criterion(matches, stopwords)
  |                   |
  |                   '----> tokenizer_2(text_1)
  |                   '----> tokenizer_2(text_2)
  |                   '----> min_distance(shared_1)
  |                   '----> min_distance(shared_2)  
  |
  |----> scissa(matches)
  |         |
  |         '----> extract_substrings(text_1, shared)
  |         '----> extract_substrings(text_2, shared)
  |         '----> compare_punctuation(text_1_substring, text_2_substring, ',')
  |         '----> compare_punctuation(text_1_substring, text_2_substring, ';')
  |         '----> compare_punctuation(text_1_substring, text_2_substring, ':') 
  |
  |----> htrg(matches)
  |         |
  |         '----> load_cracovia(cracovia_path)
  |         '----> tag_text(text_1, model, cracovia_tokenizer) 
  |                   |
  |                   '----> preprocess_text(text)
  |                   '----> group_subwords_to_words(tokens)
  |         |
  |         '----> tag_text(text_2, model, cracovia_tokenizer) 
  |                   |
  |                   '----> preprocess_text(text)
  |                   '----> group_subwords_to_words(tokens) 
  |
  |----> similarity_filter(matches)
  |         |
  |         '----> load_latincy()
  |         '----> cossim(embedding) 
  |
  '----> write_to_excel(matches, output_excel) 
  """
def parse_args():
    parser = argparse.ArgumentParser()

    # Input file 1 path
    parser.add_argument('--input_1', type=str,required=True,help='Path to the input file 1 (mandatory)')
    # Input file 2 path
    parser.add_argument('--input_2', type=str, required=True, help='Path to the input file 2 (mandatory)')
    # Genre choices
    parser.add_argument('--genre_1', required=True, choices=['prose', 'poetry'], help='Choose between "prose" and "poetry" depending on the FIRST input text.')
    parser.add_argument('--genre_2', required=True, choices=['prose', 'poetry'], help='Choose between "prose" and "poetry" depending on the SECOND input text.')
    # Stoplist
    parser.add_argument('--stoplist', type=str, default='data/stoplist.txt', help='Path to the stoplist.')
    # Skip HTRG filter
    parser.add_argument('--htrg', type=bool, default=True, help='Setting this to False skips the HTRG filter, e.g. if you cant install Cracovia system. This improves speed by a lot but expect about 25-50% more irrelevant findings.')
    # Skip similarity filter
    parser.add_argument('--similarity', type=bool, default=True, help='Setting this to False skips the similarity filter, e.g. if you cant install LatinCy. Expect about 20-30% more irrelevant findings.')

    return parser.parse_args()

def tokenizer(text: str) -> list:
    """
    Tokenizes the given text by splitting it into words based on specified delimiters.
    
    Parameters:
        text (str): The text to tokenize.
    
    Returns:
        list: A list of words extracted from the text.
    """
    # Include punctuation marks as separate tokens
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def read_in_text(input_text: str) -> List[List[str]]:
    """
    Reads the input text file and returns its content as a list of rows.
    
    Parameters:
        input_text (str): Path to the input text file.
    
    Returns:
        List[List[str]]: A list containing the rows of the input file.
    """
    with open(input_text, "r", newline='', encoding='utf-8') as samplefile: 
        reader = csv.reader(samplefile, delimiter="\t", quoting=csv.QUOTE_NONE) 
        my_list = list(reader)
    return my_list

def transform_token(token: str)-> str:
    """
    Transforms a token based on predefined prefix rules.
    
    Parameters:
        token (str): The token to transform.
    
    Returns:
        str: The transformed token.
    """
    # Dictionary mapping prefixes to their transformations
    prefix_map = {
    'adt': 'att', 'Adt': 'Att', 'adp': 'app', 'Adp': 'App', 'adc': 'acc', 'Adc': 'Acc',
    'adg': 'agg', 'Adg': 'Agg', 'adf': 'aff', 'Adf': 'Aff', 'adl': 'all', 'Adl': 'All',
    'adr': 'arr', 'Adr': 'Arr', 'ads': 'ass', 'Ads': 'Ass', 'adqu': 'acqu', 'Adqu': 'Acqu',
    'inm': 'imm', 'Inm': 'Imm', 'inl': 'ill', 'Inl': 'Ill', 'inr': 'irr', 'Inr': 'Irr',
    'inb': 'imb', 'Inb': 'Imb', 'conm': 'comm', 'Conm': 'Comm', 'conl': 'coll', 'Conl': 'Coll',
    'conr': 'corr', 'Conr': 'Corr', 'conb': 'comb', 'Conb': 'Comb', 'conp': 'comp', 'Conp': 'Comp'}
    vowels = 'aeiou'
    # Determine if the token starts with a recognized prefix
    for prefix, replacement in prefix_map.items():
        if token.lower().startswith(prefix):
            # Special handling for 'ads'/'Ads'
            if prefix.lower() == 'ads' and len(token) > 3:
                if token[3].lower() in vowels:
                    # Only apply replacement if followed by a vowel
                    new_start = token[0] + replacement[1:]  # Assimilate (eg. adserit > asserit)
                    return new_start + token[3:]
                else:
                    new_start = token[0] + token[2:] # Keep simple consonant before consonant (eg. adstringit > astringit)
                    return new_start
            elif prefix.lower() != 'ads':  # Apply other transformations directly
                new_start = token[0] + replacement[1:]  # Maintain case of the first letter
                return new_start + token[len(prefix):]
            break  # Exit the loop after finding the first matching prefix
    return token  # Return the token unchanged if no prefix matches

def assimilate(my_list: List[List[str]]) -> List[List[str]]:
    """
    Applies assimilation rules to each sentence in the list.
    
    Parameters:
        my_list (List[List[str]]): A list of sentences to process.
    
    Returns:
        List[List[str]]: The list with assimilated sentences.
    """
    for text in my_list:
        tokens = tokenizer(text[1])  # Tokenize the text
        for i, token in enumerate(tokens):
            transformed_token = transform_token(token)
            if transformed_token:
                neu = text[1].replace(tokens[i],transformed_token)
                text[1] = neu 
    return my_list

def normalize_quotation_marks(text_list: List[List[str]]) -> List[List[str]]:
    """
    Replace all types of quotation marks in each text string with a standard apostrophe.

    Args:
    - text_list: A list of lists, where each inner list contains an ID and a text string.

    Returns:
    - A new list of lists with quotation marks normalized in the text strings.
    """
    new_text_list = [[sublist[0], re.sub("[\"”“„’‘]", "\'", sublist[1])] for sublist in text_list]
    return new_text_list

# Connect -que and -ve with their preceding words
def remove_whitespace_before_connectors(text_list: List[List[str]]) -> List[List[str]]:
    """
    Remove whitespace before '-que' and '-ve' in each text string.

    Args:
    - text_list: A list of lists, where each inner list contains an ID and a text string.

    Returns:
    - A new list of lists with the whitespace removed before connectors in the text strings.
    """
    new_text_list = [[sublist[0], re.sub(" (que|ve|ue)([ ,\.!?])", r"\1\2", sublist[1])] for sublist in text_list]
    return new_text_list

# Remove digits (left over chapter numbers) at the start of each unit
def remove_left_over_counts(text_list: List[List[str]]) -> List[List[str]]:
    """
    Remove digits (leftover chapter numbers) at the start of each text string.

    Args:
    - text_list: A list of lists, where each inner list contains an ID and a text string.

    Returns:
    - A new list of lists with the leading digits removed from the text strings.
    """
    new_text_list = [[sublist[0], re.sub('\A\d{1,3}\.?', "", sublist[1])] for sublist in text_list]
    return new_text_list
    
def strip_whitespaces(text_list: List[List[str]]) -> List[List[str]]:
    """
    Remove leading and trailing whitespaces from the text strings in each sublist.

    Args:
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        A new list of sublists with leading and trailing whitespaces removed from the text strings.
    """
    new_text_list = [[sublist[0], sublist[1].strip()] for sublist in text_list]
    return new_text_list

def separate_text_at_ellipsis(text_list: List[List[str]]) -> List[List[str]]:
    """
    Separate text at ellipsis ('...') points, creating new sublists for text following the ellipsis. If an ellipsis is followed by an apostrophe ("...'"), it includes the apostrophe in the separation.

    Args:
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        The modified list with additional sublists created for text following ellipsis points.
    """
    k = 0
    while k < len(text_list):
        sublist = text_list[k]
        if '...' in sublist[1]:
            i1 = sublist[1].index('...')
            if sublist[1][i1:i1+4] == "...\'":
                text_list.insert(k+1, [sublist[0], sublist[1][i1+4:]])
                sublist[1] = sublist[1][:i1+4]
            else:
                text_list.insert(k+1, [sublist[0], sublist[1][i1+3:]])
                sublist[1] = sublist[1][:i1+3]
        k = k+1
    return text_list

def separate_text_at_punctuation_marks(text_list: List[List[str]]) -> List[List[str]]:
    """
    Separate text at specific punctuation marks (., ?, !, and : '), creating new sublists for text following these marks. This function takes care to not split ellipsis ('...').

    Args:
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        The modified list with additional sublists created for text following the specified punctuation marks.
    """
    k = 0
    while k < len(text_list):
        sublist = text_list[k]
        if '.' in sublist[1] or '?' in sublist[1] or '!' in sublist[1] or ': \'' in sublist[1]:
            if '.' in sublist[1]:
                i1 = sublist[1].index('.')
                if sublist[1][i1:i1+3] == '...': # Make sure elipsis does not get split
                    i1= len(sublist[1])+2
            else:
                i1 = len(sublist[1])+2
            if '?' in sublist[1]:
                i2 = sublist[1].index('?')
            else:
                i2 = len(sublist[1])+2
            if '!' in sublist[1]:
                i3 = sublist[1].index('!')
            else:
                i3 = len(sublist[1])+2
            if ': \'' in sublist[1]:
                i4 = sublist[1].index(': \'')
            else:
                i4 = len(sublist[1])+2
            l = min(i1, i2, i3, i4)
            text_list.insert(k+1, [sublist[0], sublist[1][l+2:]])
            sublist[1] = sublist[1][:l+2]
        k = k+1
    return text_list

def cleanup(input_list: List[List[str]]) -> List[List[str]]:
    """
    Remove sublists with empty or whitespace-only text strings, and strip leading and trailing whitespaces from remaining text strings.

    Args:
        input_list: A list of sublists, each containing an ID and a text string.

    Returns:
        A cleaned list of sublists with whitespaces stripped and empty or whitespace-only text strings removed.
    """
    output_list = [[sublist[0], sublist[1].strip()] for sublist in input_list if len(sublist) > 0 and len(sublist[1]) > 1]
    return output_list

def apply_condition_and_action(text_list: List[List[str]], condition_func: callable, action_func: callable) -> List[List[str]]:
    """
    Apply a condition and action to each pair of consecutive sublists in the text list.

    Args:
    - text_list: A list of lists, each containing an ID and a text string.
    - condition_func: A function that takes two sublists and returns True if the action should be applied.
    - action_func: A function that applies a transformation based on the condition, modifying the text list in place.

    Returns:
    - The modified text list after applying the condition and action functions.
    """
    k = 0
    while k < len(text_list)-1:  # Adjust to prevent index out of range for operations involving the next element
        if condition_func(text_list[k], text_list[k+1]):
            action_func(k, text_list)
        k += 1
    return text_list

def mark_verse_endings(text_list: List[List[str]]) -> List[List[str]]:
    """
    Append a marker (' /') at the end of each text string in the list to denote verse endings.

    Args:
        text_list: A list of sublists, each containing an ID and a text string representing a verse.

    Returns:
        A new list of sublists where each text string is appended with ' /' to denote the end of a verse.
    """
    new_text_list = [[sublist[0], sublist[1] + ' /'] for sublist in text_list]
    return new_text_list

def remove_left_over_marks(text_list: List[List[str]]) -> List[List[str]]:
    """
    Remove left over marks ('/ ') from the beginning of each text string in the list.

    Args:
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        A new list of sublists where each text string starting with '/ ' has had those initial characters removed.
    """
    new_text_list = [[sublist[0], sublist[1][2:] if sublist[1][:2] == '/ ' else sublist[1]] for sublist in text_list ]
    return new_text_list

def condition_contract_verse_endings(sublist: List[str], next_sublist: List[str]) -> bool:
    """
    Determine if the ending of a verse (' /') should lead to a contraction with the subsequent verse, excluding cases where the marker is part of a specific pattern (': /').

    Args:
        sublist: The current sublist containing an ID and a text string representing a verse.
        next_sublist: The subsequent sublist containing an ID and a text string representing the next verse.

    Returns:
        A boolean value: True if the current verse ends with ' /' (excluding ': /') and should be contracted with the next, False otherwise.
    """
    return sublist[1][-2:] == ' /' and not sublist[1][-3:] == ': /'

def condition_insertions_in_brackets(sublist: List[str], next_sublist: List[str]) -> bool:
    """
    Check if a text contains an opening bracket '(' without a corresponding closing bracket ')' afterwards. This function is used to identify cases where text spans multiple sublists due to an unclosed bracket.

    Args:
    - sublist: A list containing an ID and a text string, where the condition is checked.
    - next_sublist: The subsequent list containing an ID and a text string, not directly used in this condition but included for consistency with the framework.

    Returns:
    - A boolean value: True if there's an opening bracket without a closing bracket, False otherwise.
    """
    return '(' in sublist[1] and ')' not in sublist[1][sublist[1].index('('):]

def condition_after_interjections(sublist: List[str], next_sublist: List[str]) -> bool:
    """
    Check if the end of the current text string contains specific interjection patterns (' a!' or ' o!').

    Args:
        sublist: The current sublist containing an ID and a text string.
        next_sublist: The subsequent sublist. This parameter is included for consistency but not used in this condition.

    Returns:
        A boolean indicating whether the current text string ends with a specified interjection pattern.
    """
    return sublist[1][-3:] in [' a!', ' o!']

def condition_contract_insertions(sublist: List[str], next_sublist: List[str]) -> bool:
    """
    Determine if the start of the next text string contains a specific insertion pattern ('—').

    Args:
        sublist: The current sublist containing an ID and a text string.
        next_sublist: The subsequent sublist containing an identifier and a text string to check for the insertion pattern.

    Returns:
        A boolean indicating whether the next text string starts with the specified insertion pattern.
    """
    return next_sublist[1][0] == '—'

def condition_direct_speeches_1(sublist: List[str], next_sublist: List[str]) -> bool:
    """
    Check for a pattern indicating the beginning of inserted direct speeches, where the current text string ends with an apostrophe and the next text string starts with certain Latin terms ('inquit', 'ait', 'dixit', 'dicit').

    Args:
        sublist: The current sublist containing an ID and a text string ending with an apostrophe.
        next_sublist: The subsequent sublist containing an ID and a text string starting with the specified Latin term.

    Returns:
        A boolean indicating the presence of a direct speech pattern across the current and next text strings.
    """
    return sublist[1][-1] == '\'' and (
        next_sublist[1].startswith('inquit') or 
        next_sublist[1].startswith('ait') or 
        next_sublist[1].startswith('dixit') or 
        next_sublist[1].startswith('dicit')
    )

def condition_direct_speeches_2(sublist: List[str], next_sublist: List[str]) -> bool:
    """
    Check for a pattern indicating the beginning of inserted direct speeches, similar to 'condition_direct_speeches_1', but for cases where the next text string starts with a comma followed by certain Latin terms (', inquit', ', ait', ', dixit', ', dicit').

    Args:
        sublist: The current sublist containing an ID and a text string ending with an apostrophe.
        next_sublist: The subsequent sublist containing an ID and a text string starting with a comma followed by a Latin term.

    Returns:
        A boolean indicating the presence of a direct speech pattern with a preceding comma across the current and next text strings.
    """
    return sublist[1][-1] == '\'' and (
        next_sublist[1].startswith(', inquit') or 
        next_sublist[1].startswith(', ait') or 
        next_sublist[1].startswith(', dixit') or 
        next_sublist[1].startswith(', dicit')
    )

def action_contract_next(k: int, text_list: List[List[str]]) -> None:
    """
    Concatenates the text of the next sublist to the current sublist with a space in between and then clears the text of the current sublist.

    Args:
        k: The index of the current sublist in the text_list.
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        None. The function modifies the list in place.
    """
    text_list[k+1] = [text_list[k][0], text_list[k][1] + ' ' + text_list[k+1][1]]
    text_list[k][1] = ''

def contract_after_personal_names(text_list: List[List[str]]) -> List[List[str]]:
    """
    Contracts text after personal names, dates, and common abbreviations by appending the text of the next sublist to the current one if the current sublist ends with specific patterns.

    Args:
        text_list: A list of sublists, each containing an identifier and a text string.

    Returns:
        The modified text list after the contraction process.
    """
    l = 0
    while l < len(text_list):
        sublist = text_list[l]
        if sublist[1][-2:] == 'A.' or sublist[1][-3:] == ' a.' or sublist[1][-3:] == ' b.' or sublist[1][-2:] == 'C.' or sublist[1][-3:] == ' c.' or sublist[1][-2:] == 'D.' or sublist[1][-3:] == ' d.' or sublist[1][-2:] == 'E.' or sublist[1][-3:] == ' e.' or sublist[1][-3:] == ' f.' or sublist[1][-2:] == 'H.' or sublist[1][-2:] == 'L.' or sublist[1][-2:] == 'M.' or sublist[1][-3:] == ' m.' or sublist[1][-2:] == 'N.' or sublist[1][-2:] == 'P.' or sublist[1][-3:] == ' p.' or sublist[1][-2:] == 'Q.' or sublist[1][-3:] == ' q.' or sublist[1][-2:] == 'S.' or sublist[1][-2:] == 'T.' or sublist[1][-3:] == ' t.' or sublist[1][-3:] == ' v.' or sublist[1][-3:] == 'An.' or sublist[1][-4:] == ' an.' or sublist[1][-3:] == 'Cn.' or sublist[1][-4:] == ' in.' or sublist[1][-4:] == ' ex.'or sublist[1][-4:] == ' ut.' or sublist[1][-3:] == 'M\'.' or sublist[1][-3:] == 'R\'.' or sublist[1][-4:] == ' pl.' or sublist[1][-3:] == 'Sp.' or sublist[1][-3:] == 'Ti.' or sublist[1][-4:] == ' tr.' or sublist[1][-3:] == 'Id.' or sublist[1][-4:] == 'App.' or sublist[1][-4:] == 'Ser.' or sublist[1][-4:] == 'Sex.' or sublist[1][-4:] == 'Tib.' or sublist[1][-5:] == ' cos.' or sublist[1][-4:] == 'Cos.' or sublist[1][-4:] == 'Kal.' or sublist[1][-5:] == ' kal.' or sublist[1][-5:] == ' med.' or sublist[1][-4:] == 'Med.' or sublist[1][-4:] == 'Non.' or sublist[1][-5:] == ' non.' or sublist[1][-5:] == ' scr.' or sublist[1][-4:] == 'Scr.' or sublist[1][-5:] == ' vid.' or sublist[1][-4:] == 'Vid.' or sublist[1][-5:] == 'Mart.' or sublist[1][-4:] == 'Apr.' or sublist[1][-4:] == 'Mai.' or sublist[1][-4:] == 'Iun.' or sublist[1][-6:] == 'Quint.' or sublist[1][-5:] == 'Sext.' or sublist[1][-5:] == 'Sept.' or sublist[1][-4:] == 'Oct.' or sublist[1][-4:] == 'Nov.' or sublist[1][-4:] == 'Dec.' or sublist[1][-4:] == 'Ian.' or sublist[1][-5:] == 'Febr.' or sublist[1][-6:] == ' coss.' or sublist[1][-5:] == 'Coss.' or sublist[1][-5:] == 'fort.' or sublist[1][-6:] == ' prid.' or sublist[1][-5:] == 'Prid.' or sublist[1][-2:] == '.,' or sublist[1][-2:] == '?)' or sublist[1][-4:] == 'frg.' or sublist[1][-6:] == 'Schol.' or sublist[1][-4:] == 'Cus.': 
            text_list[l+1] = [sublist[0], sublist[1] + ' ' + text_list[l+1][1]]
            sublist.clear()    
        l = l+1
    return text_list

def write_output(text_list: List[List[str]], output_path: str) -> None:
    """
    Writes the content of text_list to a txt file at the specified output path.

    Args:
        text_list: A list of sublists, each containing an identifier and a text string.
        output_path: The file path where the output txt file will be written.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        result = csv.writer(file, delimiter='\t', quotechar='|')
        for i in text_list:
            if len(i) > 1:
                result.writerow(i)

def assimilation(input_text: str) -> List[List[str]]:
    """
    Processes the input text through assimilation steps.

    Args:
        input_text: The path to the input text file to be assimilated.

    Returns:
        A list of sublists, each containing an ID and the assimilated text string.
    """
    text_list = read_in_text(input_text)
    assimilated_list = assimilate(text_list)
    return assimilated_list

def limited_bracket_contraction(text_list: List[List[str]]) -> List[List[str]]:
    """
    Contracts unmatched '(' with the next 1 or 2 entries if a closing ')' appears.
    If no closing ')' is found within 2 elements, leaves the original sublists unchanged.

    Returns:
        Modified text_list with limited-range bracket contractions.
    """
    k = 0
    while k < len(text_list) - 1:
        current = text_list[k][1]
        if '(' in current and ')' not in current[current.index('('):]:
            # Look ahead max 2 items for a closing bracket
            max_lookahead = min(2, len(text_list) - k - 1)
            combined = current
            found_closing = False
            for i in range(1, max_lookahead + 1):
                combined += ' ' + text_list[k + i][1]
                if ')' in text_list[k + i][1]:
                    found_closing = True
                    break
            if found_closing:
                # Contract only up to and including the match
                text_list[k + i][1] = combined
                for j in range(i):
                    text_list[k + j][1] = ''
                k += i  # skip over the modified range
            else:
                k += 1  # no match found, do nothing
        else:
            k += 1
    return text_list

def merge_if_no_terminal_punctuation(text_list: List[List[str]]) -> List[List[str]]:
    """
    Merges lines that do not end with a terminal punctuation mark by appending the next line's content
    and copying the current metadata to the next line. The current line's text is cleared.

    Args:
        text_list: A list of [metadata, text] sublists.

    Returns:
        The modified list with merged text and updated metadata.
    """
    k = 0
    while k < len(text_list) - 1:
        current_text = text_list[k][1].strip()
        # Check if the last character is not terminal punctuation
        if not current_text or current_text[-1] not in '.!?:':
            # Merge text with the next
            text_list[k + 1][1] = current_text + ' ' + text_list[k + 1][1].lstrip()
            text_list[k + 1][0] = text_list[k][0]  # Copy metadata
            text_list[k][1] = ''  # Clear current line's text
        k += 1
    return text_list

def phrasing_prose(text_list: List[List[str]]) -> List[List[str]]:
    """
    Processes a list of text elements for prose by applying a series of normalization and cleanup steps, including normalization of quotation marks, removal of whitespace
    around connectors, stripping of leading chapter numbers and whitespace, and separation of text at ellipsis and punctuation marks. Conditions and actions are applied to handle
    specific formatting cases.

    Args:
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        The processed list of text elements after applying all normalization and cleanup steps.
    """
    text_list = normalize_quotation_marks(text_list)
    text_list = remove_whitespace_before_connectors(text_list)
    text_list = remove_left_over_counts(text_list)
    text_list = strip_whitespaces(text_list)
    text_list = separate_text_at_ellipsis(text_list)
    text_list = separate_text_at_punctuation_marks(text_list)
    text_list = cleanup(text_list)
    text_list = limited_bracket_contraction(text_list)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_after_interjections, action_contract_next)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_contract_insertions, action_contract_next)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_direct_speeches_1, action_contract_next)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_direct_speeches_2, action_contract_next)
    text_list = cleanup(text_list)
    text_list = contract_after_personal_names(text_list)
    text_list = cleanup(text_list)
    text_list = merge_if_no_terminal_punctuation(text_list)
    text_list = cleanup(text_list)
    #write_output(text_list, output_path)
    return text_list

def phrasing_poetry(text_list: List[List[str]]) -> List[List[str]]:
    """
    Processes a list of text elements for poetry by marking verse endings, applying a series of normalization and cleanup steps similar to prose, but with additional handling for verse
    endings.

    Args:
        text_list: A list of sublists, each containing an ID and a text string.

    Returns:
        The processed list of poetry text elements after applying all normalization, cleanup,
        and verse ending marking steps.
    """
    text_list = mark_verse_endings(text_list)
    text_list = normalize_quotation_marks(text_list)
    text_list = remove_whitespace_before_connectors(text_list)
    text_list = remove_left_over_counts(text_list)
    text_list = strip_whitespaces(text_list)
    text_list = separate_text_at_ellipsis(text_list)
    text_list = separate_text_at_punctuation_marks(text_list)
    text_list = apply_condition_and_action(text_list, condition_contract_verse_endings, action_contract_next)
    text_list = cleanup(text_list)
    text_list = limited_bracket_contraction(text_list)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_after_interjections, action_contract_next)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_contract_insertions, action_contract_next)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_direct_speeches_1, action_contract_next)
    text_list = cleanup(text_list)
    text_list = apply_condition_and_action(text_list, condition_direct_speeches_2, action_contract_next)
    text_list = cleanup(text_list)
    text_list = contract_after_personal_names(text_list)
    text_list = cleanup(text_list)
    text_list = remove_left_over_marks(text_list)
    text_list = cleanup(text_list)
    text_list = merge_if_no_terminal_punctuation(text_list)
    text_list = cleanup(text_list)
    write_output(text_list, "phrasing_vergil.txt")
    return text_list

def maketrans(x: str, y: str) -> Dict[int, str]:
    """
    Create a translation table used for string replacements.

    Args:
        x (str): Characters to replace.
        y (str): Characters to replace with.

    Returns:
        Dict[int, str]: A translation table compatible with str.translate().
    """
    return str.maketrans(x,y)

def normalize_orthographic_variants(text_list: List[List[str]]) -> List[List[str]]:
    """
    Normalize orthographic variants in a list of text strings by replacing specific characters.

    Args:
        text_list (List[List[str]]): A nested list of strings to normalize.

    Returns:
        List[List[str]]: The text list with normalized orthographic variants.
    """
    intab = "vjVJ"
    outtab = "uiUI"
    transtab = maketrans(intab,outtab)    
    new_text_list = [[item.translate(transtab) for item in sublist] for sublist in text_list]
    return new_text_list

def normalize_typography(text_list: List[List[str]]) -> List[List[str]]:
    """
    Normalize typography in a list of text strings, setting lowercase and removing whitespace.

    Args:
        text_list (List[List[str]]): A nested list of strings to normalize.

    Returns:
        List[List[str]]: The text list with normalized typography.
    """
    r = re.compile(r'\xe2\x80\x9c|\xe2\x80\x9d|\x9c|\x9d|\xef\xbb\xbf')
    new_text_list = [[item.lower().strip() for item in sublist] for sublist in text_list]
    new_text_list = [[' ' + item + ' ' for item in sublist] for sublist in new_text_list]
    new_text_list = [[r.sub('', item) for item in sublist] for sublist in new_text_list]
    return new_text_list

def remove_unwanted_characters(text_list: List[List[str]]) -> List[List[str]]:
    """
    Remove unwanted punctuation and whitespace characters from a list of text strings.

    Args:
        text_list (List[List[str]]): A nested list of strings from which to remove characters.

    Returns:
        List[List[str]]: The text list with unwanted characters removed and excessive spaces reduced.
    """
    intab2 = "—?!-,.()[]:;\'\/\"”“„"  
    outtab2 = "                   "
    transtab2 = maketrans(intab2,outtab2)
    new_text_list = [[item.translate(transtab2) for item in sublist] for sublist in text_list]
    new_text_list = [[re.sub(" {2,}", " ", item) for item in sublist] for sublist in text_list]
    return new_text_list

def tokenizer_2(text: str) -> List[str]:
    """
    Simple tokenizer that splits text into words based on specified punctuation characters.

    Args:
        text (str): The string to tokenize.

    Returns:
        List[str]: A list of words from the input string.
    """
    wort = ''
    worttrennung = ' —?!-,.()[]:;\'\/\"”“„'
    listetext = []
    text = text + '.'
    for k in text:
        if k not in worttrennung:
            wort = wort + k
        else:
            if len(wort) > 0:
                listetext.append(wort)
                wort = ''
    return listetext

def tokenize_text(text_list: List[List[str]]) -> List[List[str]]:
    """
    Tokenize each string in a nested list of text strings using a custom tokenizer.

    Args:
        text_list (List[List[str]]): A nested list of strings to tokenize.

    Returns:
        List[List[str]]: A nested list where each inner list contains tokens from the corresponding original string.
    """
    tokens = []
    for sublist in text_list:
            for i in sublist:
                tokenisiert = tokenizer_2(i)
                tokens.append(tokenisiert)
    return tokens
            
def initialize_stopwords(filename: str) -> set:
    """
    Reads a list of stopwords from a given file where each stopword is on a new line.

    Args:
        filename (str): The path to the file containing stopwords.

    Returns:
        Set[str]: A set containing the stopwords.
    """
    stopwords = set()
    with open(filename, "r", newline='', encoding='utf-8') as file: 
        reader = csv.reader(file, delimiter = "\t", quoting = csv.QUOTE_NONE) 
        stoplist = list(reader)
    for item in stoplist:
            if item:  # Check if the item list is not empty
                stopwords.add(item[0].strip())
    return stopwords

def highlight_shared_words(sentence: str, matched_items: list) -> str:
    """
    Highlights words in a sentence that are found in a list of matched items.

    Args:
        sentence (str): The sentence in which to highlight words.
        matched_items (list): A list of words that should be highlighted within the sentence.

    Returns:
        str: The modified sentence with matched words surrounded by double asterisks ('**') to denote highlighting.
    """
    trennung = ' —?!-,.()[]:;\'\/\"”“„'
    wort = ''
    for k in sentence:
        if k not in trennung:
            wort = wort + k
        else:
            if wort.lower() in matched_items:
                for t1 in trennung:
                    for t2 in trennung:
                        sentence = sentence.replace(t1 + wort + t2, t1 + '**' + wort + '**' + t2)
            wort = ''
    sentence = sentence.replace('****', '**') #if four stars in a row occured, replace them
    sentence = sentence.strip()
    return sentence

def find_four_adjacent(numbers: List[int]) -> List[int]:
    """
    Checks if there are at least four adjacent integers in a list.

    Args:
        numbers (list): A list of integers representing token indices.

    Returns:
        list: The longest sequence of at least four adjacent numbers found.
    """
    sorted_numbers = sorted(set(numbers))
    count = 1  # We start counting with the first number in the sorted list
    adjacent_sequence = [sorted_numbers[0]]  # Start with the first number in the sequence

    sequence_storage = []
    for i in range(1, len(sorted_numbers)):
        if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
            count += 1
            adjacent_sequence.append(sorted_numbers[i])
            # Check if we have found at least four adjacent numbers
            if count >= 4:
                sequence_storage.append(adjacent_sequence)
        else:
            # Reset count and sequence when a break in adjacency is found
            count = 1
            adjacent_sequence = [sorted_numbers[i]]

    if sequence_storage:
        return max(sequence_storage, key=len)  # Return the longest sequence found
    else:
        return []  # Return an empty list if no sequence of four or more adjacent numbers was found

def text_matching(tokens_1: List[List[str]], tokens_2: List[List[str]], original_text_list_1: List[List[str]], original_text_list_2: List[List[str]], meta_list_1: List[str], meta_list_2: List[str], stopwords: Set[str]) -> List[List]:
    """
    Matches tokens between two lists of tokenized texts, excluding stopwords, and highlights matches in the texts.
    Beside the regular text matching a second matching process takes place, which we call the Complura-Filter: Ignoring stopwords it looks for at least 4 (default, adjustable global var) adjacent 
    tokens among both texts. These findings are considered relevant and directly written to a separate Excel file.

    Args:
        tokens_1 (List[List[str]]): Tokenized text from the first document.
        tokens_2 (List[List[str]]): Tokenized text from the second document.
        original_text_list_1 (List[List[str]]): Original texts of the first document.
        original_text_list_2 (List[List[str]]): Original texts of the second document.
        meta_list_1 (List[str]): Metadata corresponding to the first document.
        meta_list_2 (List[str]): Metadata corresponding to the second document.
        stopwords (Set[str]): Set of stopwords to exclude from matching.

    Returns:
        List[List]: A list of matches with metadata and highlighted texts.
    """
    matches = []
    complura_matches = []
    count= 0
    complura_count = 0
    for index_a in range(len(tokens_1)):
        sublist_a = tokens_1[index_a]
        for index_b in range(len(tokens_2)):
            sublist_b = tokens_2[index_b]
            matched_items = []
            complura_matched_items = []
            for position_a, item_a in enumerate(sublist_a):
                for position_b, item_b in enumerate(sublist_b):
                    if item_a == item_b and item_a:
                        complura_matched_items.append((item_a, position_a, position_b))
                    if item_a == item_b and item_a not in matched_items:
                        if item_a not in stopwords:
                            matched_items.append(item_a)

            # This block represents the Complura-Filter: It looks for at least 4 (default value, adjustable) adjacent tokens that are shared among both sentences, in contrast to the rest including stopwords.
            if len(complura_matched_items) >= min_number_complura:
                indices_a = [element[1] for element in complura_matched_items]
                indices_b = [element[2] for element in complura_matched_items]
                # Find longest adjacent sequence of indices
                index_sequence_a = find_four_adjacent(indices_a)
                index_sequence_b = find_four_adjacent(indices_b)
                
                if len(index_sequence_a) >= min_number_complura and len(index_sequence_b) >= min_number_complura:
                    # Convert indices back into words
                    complura_shared = [sublist_a[idx] for idx in index_sequence_a]
                    complura_count += 1
                    complura_listesatz_a = original_text_list_1[index_a] #get line as a list out of the original text
                    complura_sentence_a = ' ' + complura_listesatz_a[0] + ' ' 
                    complura_listesatz_b = original_text_list_2[index_b]
                    complura_sentence_b = ' ' + complura_listesatz_b[0] + ' ' 
                    
                    # Highlight the shared words
                    complura_sentence_a = highlight_shared_words(complura_sentence_a, complura_shared)
                    complura_sentence_b = highlight_shared_words(complura_sentence_b, complura_shared)

                    # Concatenate the shared words into a ; separated string for the output
                    complura_shared_words = ''
                    for element in complura_shared: 
                        if len(complura_shared_words)>0:
                            complura_shared_words = complura_shared_words + '; ' + element
                        else:
                            complura_shared_words = element
                    
                    # Append to complura_matches
                    complura_line = [complura_count, meta_list_1[index_a]] + [complura_sentence_a] + [meta_list_2[index_b]] +  [complura_sentence_b] + [complura_shared_words]
                    complura_matches.append(complura_line)

            # Here the regular matches get processed
            if len(matched_items) >= min_number_of_shared_words:
                count = count + 1 
                listesatz_a = original_text_list_1[index_a] #get line as a list out of the original text
                sentence_a = ' ' + listesatz_a[0] + ' ' 
                listesatz_b = original_text_list_2[index_b]
                sentence_b = ' ' + listesatz_b[0] + ' '

                # Highlight the shared words
                sentence_a = highlight_shared_words(sentence_a, matched_items)
                sentence_b = highlight_shared_words(sentence_b, matched_items)
                shared_words = ''
                for element in matched_items:
                    if len(shared_words)>0:
                        shared_words = shared_words + '; ' + element
                    else:
                        shared_words = element
                line = [count, meta_list_1[index_a]] + [sentence_a] + [meta_list_2[index_b]] +  [sentence_b] + [shared_words]
                matches.append(line)

    # Prepare the Complura matches and write them to Excel            
    # complura_matches_meta = [['Number', 'Passage Text A', 'Text A', 'Passage Text B', 'Text B', 'shared words']] + complura_matches
    # write_to_excel(complura_matches_meta, output_complura)

    # Return the matches
    return matches, complura_matches
    

def separate_text_meta(text_list_1: List[List[str]], text_list_2: List[List[str]]) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Separates text and metadata for two documents stored in nested lists.

    Args:
        text_list_1 (List[List[str]]): Nested list containing text and metadata for the first document.
        text_list_2 (List[List[str]]): Nested list containing text and metadata for the second document.

    Returns:
        Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]: Separated text and metadata for both documents.
    """
    text_1_meta = []
    text_2_meta = []
    text_1 = []
    text_2 = []
    for sublist in text_list_1:
        text_1_meta.append(sublist[0])
        text_1.append([sublist[1]])
    for sublist in text_list_2:
        text_2_meta.append(sublist[0])
        text_2.append([sublist[1]])
    return text_1, text_1_meta, text_2, text_2_meta

def min_distance(shared_tokens: List[Tuple[str, int]]) -> float:
    """
    Calculates the minimum distance between indices of shared tokens.

    Args:
        shared_tokens (List[Tuple[str, int]]): A list of tuples containing tokens and their indices.

    Returns:
        float: The minimum distance between any two shared tokens, or float('inf') if no pairs exist.
    """
    indices = [index for token, index in shared_tokens]
    return min([abs(indices[i] - indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))], default=float('inf'))

def apply_distance_criterion(matches: List[List[str]], stopwords: set) -> List[List[str]]:
    """
    Filters matches based on the distance criterion, where tokens' maximum distance must be equal to the global variable 'maximum_distance_between_shared_words' or less, stopwords excluded.

    Args:
        matches (List[List[str]]): List of text lines to be compared.
        stopwords (set): A set of stopwords to be excluded from the distance calculation.

    Returns:
        List[List[str]]: Filtered list of matches meeting the distance criteria.
    """
    new_matches = []
    for line in matches:
        text_1 = line[2]
        text_2 = line[4]
        tokens_1 = tokenizer_2(text_1)
        tokens_2 = tokenizer_2(text_2)
        tokens_1 = [token for token in tokens_1 if token not in stopwords]
        tokens_2 = [token for token in tokens_2 if token not in stopwords]
        shared_1 = [(token, index) for index, token in enumerate(tokens_1) if token.startswith("**")]
        shared_2 = [(token, index) for index, token in enumerate(tokens_2) if token.startswith("**")]

        # Calculate minimum distances
        distance_1 = min_distance(shared_1)
        distance_2 = min_distance(shared_2)
        
        # Check if both distances meet the criterion
        if distance_1 <= maximum_distance_between_shared_words and distance_2 <= maximum_distance_between_shared_words:
            new_matches.append(line)

    return new_matches

def write_matches(matches_meta: List[List[str]], output_file: str) -> None:
    """
    Writes matching results to an ouput file (tab separated).
    
    Args:
        matches_meta (List[List[str]]): List of text lines to be written to a file.
        output_file (str): Path to file. 
    """
    with open (output_file, 'w', newline='', encoding = 'utf-8') as file:
        result = csv.writer(file, delimiter = '\t', quotechar = '|') 
        for i in matches_meta:
            result.writerow(i)

def compare_texts(text_list_1: List[List[str]], text_list_2: List[List[str]], stop_list: str) -> List[List[str]]:
    """
    Control function for the text matching. The preprocessed texts are read in and split in text and metadata. 
    The text is further normalized and tokenized. Text matching is done sentence by sentence: If a sentence-pair has at least X shared words (global variable 'min_number_shared_words') stopwords excluded,
    it is evaluated as a potential match. Next the distance between the shared words is calculated. When the shared words are too far apart (global variable 'maximum_distance_between_shared_words')
    stopwords excluded, the finding is considered irrelevant and not retained.

    Args:
        text_list_1 (List[List[str]]): The first list of text.
        text_list_2 (List[List[str]]): The second list of text.
        stop_list (str): The filename containing stopwords.

    Returns:
        List[List[str]]: A list of matches with additional metadata.
    """
    pure_text_list_1, meta_list_1, pure_text_list_2, meta_list_2 = separate_text_meta(text_list_1, text_list_2)
    # Orthography i/j and v/u (kept in "original" files)
    pure_text_list_1 = normalize_orthographic_variants(pure_text_list_1)
    pure_text_list_2 = normalize_orthographic_variants(pure_text_list_2)
    # Set original text lists for later
    original_text_list_1 = pure_text_list_1[:]
    original_text_list_2 = pure_text_list_2[:]
    # Normalization steps
    pure_text_list_1 = normalize_typography(pure_text_list_1)
    pure_text_list_1 = remove_unwanted_characters(pure_text_list_1)
    pure_text_list_2 = normalize_typography(pure_text_list_2)
    pure_text_list_2 = remove_unwanted_characters(pure_text_list_2)
    # Tokenization
    tokens_1 = tokenize_text(pure_text_list_1)
    tokens_2 = tokenize_text(pure_text_list_2)
    # initialize Stopwords
    stopwords = initialize_stopwords(stop_list)
    # actual text matching
    matches, matches_complura = text_matching(tokens_1, tokens_2, original_text_list_1, original_text_list_2, meta_list_1, meta_list_2, stopwords)

    # Add descriptions of the columns and write matches to file
    # matches_meta = [['Number', 'Passage Text A', 'Text A', 'Passage Text B', 'Text B', 'shared words']] + matches 
    # write_matches(matches_meta, output_matches)

    # Check for the distance criterion
    matches = apply_distance_criterion(matches, stopwords)
    # matches_meta = [['Number', 'Passage Text A', 'Text A', 'Passage Text B', 'Text B', 'shared words']] + matches 
    # write_matches(matches_meta, output_distance)
            
    return matches, matches_complura

def extract_substrings(text: str, shared: List[str]) -> List[str]:
    """
    Extracts substrings from the provided text between markers defined by keywords in 'shared'.
    
    Args:
        text (str): The text from which substrings are to be extracted.
        shared (List[str]): List containing two keywords which define the boundaries for substring extraction.

    Returns:
        List[str]: A list of substrings extracted from the text between the specified keywords.
    """
    collection = []
    text_lower = text.lower()
    shared = [s.lower() for s in shared]

    while '**' + shared[0].strip() + '**' in text_lower and '**' + shared[1].strip() + '**' in text_lower:
        if text_lower.index('**' + shared[0].strip() + '**') < text_lower.index('**' + shared[1].strip() + '**'):
            first = '**' + shared[0].strip() + '**'
            second = '**' + shared[1].strip() + '**'
        else:
            first = '**' + shared[1].strip() + '**'
            second = '**' + shared[0].strip() + '**'

        start = text_lower.index(first)
        end = text_lower.index(second)
        collection.append(text[start + len(first):end])

        # Update text to find next occurrence
        text_lower = text_lower[end + len(second):]
        text = text[end + len(second):]

    return collection


def compare_punctuation(text_1_substring: List[str], text_2_substring: List[str], punctuation: str) -> List[int]:
    """
    Compares the count of a specific punctuation mark in substrings from two lists.

    Args:
        text_1_substring (List[str]): List of substrings from the first text.
        text_2_substring (List[str]): List of substrings from the second text.
        punctuation (str): The punctuation mark to count.

    Returns:
        List[int]: List of comparison results; 1 if counts match, 0 otherwise.
    """
    results = []
    for hier in text_1_substring:
        for quell in text_2_substring:
            if hier.count(punctuation) == quell.count(punctuation):
                results.append(1)
            else:
                results.append(0)
    return results


def scissa(matches: List[List[str]]) -> List[List[str]]:
    """
    Control function for the scissa filter. Characters between shared words are extracted and punctuations (,;;) counted.
    If the number of all three punctuation marks is not equal in both sentences, the finding is considered irrelevant and not retained.

    Args:
        matches (List[List[str]]): List of text matches to process.

    Returns:
        List[List[str]]: Filtered matches after applying the comparison criteria.
    """
    new_matches = []
    for line in matches:
        shared = line[5].split(";")
        if len(shared) == 2:
            text_1 = line[2]
            text_2 = line[4]

            # Extracting substrings
            text_1_substring = extract_substrings(text_1, shared)
            text_2_substring = extract_substrings(text_2, shared)

            # Comparing punctuation
            commas = compare_punctuation(text_1_substring, text_2_substring, ',')
            semicolons = compare_punctuation(text_1_substring, text_2_substring, ';')
            colons = compare_punctuation(text_1_substring, text_2_substring, ':')

            # Filter non matching findings
            if all(x == 1 for x in commas) and all(x == 1 for x in semicolons) and all(x == 1 for x in colons):
                new_matches.append(line)
        elif len(shared) >= 3:
            new_matches.append(line)
    #matches_meta = [['Number', 'Passage Text A', 'Text A', 'Passage Text B', 'Text B', 'shared words']] + new_matches 
    #write_matches(matches_meta, output_scissa)

    return new_matches

def load_cracovia() -> Language:
    """
    Load the tokenizer and model from the specified path (global variable).

    Returns:
        tuple: A tuple containing the tokenizer and model.
    """
    # Load the model and tokenizer
    cracovia_tokenizer = XLMRobertaTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = XLMRobertaForTokenClassification.from_pretrained(BASE_MODEL_NAME)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model loaded on CUDA")
    else:
        print("CUDA not available, using CPU")

    model.eval()
    return cracovia_tokenizer, model

# Preprocessing function; punctuation and asterisks
def preprocess_text(text: str) -> str:
    """
    Prepares text for tokenization by spacing punctuation and adding a dummy word at the beginning.

    Args:
        text (str): The original text string.

    Returns:
        str: The preprocessed text string.
    """
    # Add spaces around punctuation so they're treated as separate tokens
    for punct in string.punctuation:
        text = text.replace(punct, f' {punct} ')
    text = " et " + text # as the model frequently tagged the first word wrong, adding a dummy latin word improved the performance on the actual first word
    return text

def group_subwords_to_words(tokenized_sentence: List[str]) -> List[str]:
    """
    Converts a list of subword tokens back into whole words.

    Args:
        tokenized_sentence (List[str]): The list of subword tokens.

    Returns:
        List[str]: A list of whole words reconstructed from subword tokens.
    """
    grouped_tokens = []
    current_word = ""
    for token in tokenized_sentence:
        if token.startswith("▁"):  # This indicates a new word in SentencePiece tokenization.
            if current_word:
                grouped_tokens.append(current_word)
            current_word = token[1:]  # skip the "▁" character
        else:
            current_word += token
    if current_word:
        grouped_tokens.append(current_word)
    return grouped_tokens

def tag_text(text: str, model, cracovia_tokenizer) -> List[Tuple[str, str]]:
    """
    Tags text with parts of speech using a pre-trained model and tokenizer.

    Args:
        text (str): The text to tag.
        model: The pre-trained POS tagging model, Cracovia system.
        cracovia_tokenizer: The tokenizer for the model.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing words and their corresponding POS tags.
    """
    # Preprocessing and Tokenization
    tokens = cracovia_tokenizer.tokenize(preprocess_text(text))

    # In some rare cases a way too large text was put together in the matching process. This block eliminates these errors that could cause the HTRG to fail
    MAX_TOKEN_LENGTH = 512
    # Skip processing if tokens exceed the maximum length
    if len(tokens) > MAX_TOKEN_LENGTH:
        print(f"Skipping text as it exceeds maximum token length: {len(tokens)} tokens")
        return []
    
    input_ids = cracovia_tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor([input_ids])

    # Move input tensor to the same device as the model
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
              
    # Convert numbers into tags
    id2label = {0: "", 1: "ADJ", 2: "ADP", 3: "ADV", 4: "AUX", 5: "CCONJ", 6: "DET", 7: "INTJ", 8: "NOUN", 9: "NUM", 10: "PART", 11: "PRON", 12: "PROPN", 13: "PUNCT", 14: "SCONJ", 15: "VERB", 16: "X", 17: "O"}

    token_tags = [id2label[id] for id in predictions]

    # Regroup the subword tokens
    grouped_tokens = group_subwords_to_words(tokens)

    # Approach to match words in grouped_tokens to their tags
    token_idx = 0
    predicted_tags = []

    for word in grouped_tokens:
        word_as_subtokens = cracovia_tokenizer.tokenize(word)
        num_subtokens = len(word_as_subtokens)
        
        # Use the tag of the first subtoken for the word
        predicted_tags.append(token_tags[token_idx])
        
        # Move to the next token
        token_idx += num_subtokens

    #Make a dict out of the results
    word_tag_list = list(zip([word.lower() for word in grouped_tokens], predicted_tags))
    return word_tag_list

def htrg(matches: List[List[str]]) -> List[List[str]]:
    """
    Control function for the HTRG filter. For further filtering the Parts-of-Speech of the shared words are analyzed by the Cracovia system.
    Only the POS tags NOUN VERB and PROPER NOUN (PROPN) are considered relevant and retained. 

    Args:
        matches (List[List[str]]): The list of text matches.

    Returns:
        List[List[str]]: The list of matches that meet specified grammatical criteria.
    """
    cracovia_tokenizer, model = load_cracovia()
    grammar = [('NOUN', 'VERB'), ('VERB', 'NOUN'), ('NOUN', 'NOUN'), ('VERB', 'VERB'), ('PROPN', 'NOUN'), ('PROPON', 'VERB'), ('NOUN', 'PROPN'), ('VERB', 'PROPN'), ('PROPN', 'PROPN')]   #HTRG als Tupel, alle Varianten vorwärts und rückwärts 
    G = set(grammar)
    new_matches = []
    for line in matches:
        shared = line[5].split(";")
        shared = [word.strip() for word in shared]
        text_1 = line[2].replace('**',' ')
        text_2 = line[4].replace('**',' ')
        word_tag_list_1 = tag_text(text_1, model, cracovia_tokenizer) 
        word_tag_list_2 = tag_text(text_2, model, cracovia_tokenizer) 
        tokens_1 = [element[0] for element in word_tag_list_1] 
        tokens_2 = [element[0] for element in word_tag_list_2] 
        tags_1 = [element[1] for element in word_tag_list_1] 
        tags_2 = [element[1] for element in word_tag_list_2] 

        pos_1 = []
        pos_2 = []
        for item in shared:
            indices_1 = [i for i,val in enumerate(tokens_1) if val.lower() == item]
            indices_2 = [i for i,val in enumerate(tokens_2) if val.lower() == item]
            shared_hier_tags = [tags_1[i] for i in indices_1]
            shared_verg_tags = [tags_2[i] for i in indices_2]
            pos_1.append(shared_hier_tags)
            pos_2.append(shared_verg_tags)
        combinations_1 = list(itertools.product(*pos_1))
        combinations_2 = list(itertools.product(*pos_2))
        line.append(combinations_1)
        tuples_1 = [set(combinations(s,2)) for s in combinations_1]
        htrg_1 = []
        for s in tuples_1:
            c = G.intersection(s)
            if len(c) > 0:
                htrg_1.append(c)
        """  if len(htrg_hier) > 0:
            line.append('HTRG: in')
        else:
            line.append('HTRG: out') """
        line.append(combinations_2)  
        tuples_2 = [set(combinations(s,2)) for s in combinations_2]
        htrg_2 = []
        for s in tuples_2:
            c = G.intersection(s)
            if len(c) > 0:
                htrg_2.append(c)
        """ if len(htrg_verg) > 0:
            line.append('HTRG: in')
        else:
            line.append('HTRG: out') """

        shared_combinations = set(combinations_1).intersection(set(combinations_2)) #Find combinations that match
        #line.append(shared_combinations)

        shared_tuples = [x.intersection(y) for x,y in zip(tuples_1,tuples_2)] #Find tuples that are the same
        #line.append(shared_tuples)
        htrg_shared = []
        for s in shared_tuples:
            c = G.intersection(s)
            if len(c) > 0:
                htrg_shared.append(c)
        if len(htrg_shared) > 0:
            line.append('in')
            new_matches.append(line)
        """else:
            line.append('out')

         if line[-6] == line[-4]:
            line.append('both ' + line[-6])
        else:
            line.append('both differ in HTRG') """
        
    #matches_meta = [['Number', 'Passage Text A', 'Text A', 'Passage Text B', 'Text B', 'shared words', "POS A", "POS B", "HTRG"]] + new_matches 
    #write_matches(matches_meta, output_htrg)
    return new_matches

def load_latincy() -> Language:
    """
    Loads a spaCy language model specified by the global variable `spacy_model`.

    Returns:
        Language: A loaded spaCy language model ready for processing text.
    """
    nlp = spacy.load(spacy_model)
    return nlp

def cossim(vectorlist: List[numpy.ndarray]) -> Union[float, str]: 
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vectorlist (List[numpy.ndarray]): A list containing two numpy arrays representing vectors.

    Returns:
        Union[float, str]: The cosine similarity as a float, or 'not defined' if any vector is empty.
    """
    if numpy.any(vectorlist[0]):
        if numpy.any(vectorlist[1]):
            cosinus = dot(vectorlist[0], vectorlist[1])/(norm(vectorlist[0])*norm(vectorlist[1]))
        else:
            cosinus = "not defined"
    else:
        cosinus = "not defined"
    return cosinus

def similarity_filter(matches: List[List[str]]) -> List[List[str]]:
    """
    Control function for the similarity filter. Filters a list of text matches based on the cosine similarity of embeddings from shared words.
    Uses a spaCy model to generate embeddings and a global threshold `similarity_threshold` to determine relevance. Output is also written as a tab-separated .txt file.

    Args:
        matches (List[List[str]]): A list of matches, each an array of strings with shared words at index 5.

    Returns:
        List[List[str]]: The filtered list of matches including similarity scores.
    """
    nlp = load_latincy()
    new_matches = []
    for line in matches:
        shared = line[5].split(";")
        shared = [word.strip() for word in shared] 
        if len(shared) == 2:
            doc_1 = nlp(shared[0])
            doc_2 = nlp(shared[1])
            embedding_1 = doc_1.vector
            embedding_2 = doc_2.vector
            embedding = [embedding_1, embedding_2]
            sim = round(cossim(embedding), 2)
            line.append(sim)
            if sim <= similarity_threshold:
                new_matches.append(line)
        else:
            new_matches.append(line)
        
    return new_matches

def write_to_excel(matches: List[List[str]], output_path) -> None:
    """
    Writes a list of matches to an Excel file.

    Args:
        matches (List[List[str]]): A list of matches to be written to an Excel file.
    """
    df = pd.DataFrame(matches)
    df.to_excel(output_path, index=False, header=False, engine='openpyxl')

def main():
    # Start the timer
    start_time = time.perf_counter()   
    
    args = parse_args()
    input_text_1 = args.input_1
    input_text_2 = args.input_2
    genre_1 = args.genre_1
    genre_2 = args.genre_2
    stop_list = args.stoplist
    htrg_switch = args.htrg
    similarity_switch = args.similarity
    text_name_combined = input_text_1[5].strip(".txt") + "_" + input_text_2[5].strip(".txt")

    global output_complura
    output_complura = text_name_combined + "_complura.xlsx"
    output_excel = text_name_combined + "_results.xlsx"

    # Assimilate both input txt files. The hypotext is text 1, the hypertext is text 2.
    assimilated_list_1 = assimilation(input_text_1)
    assimilated_list_2 = assimilation(input_text_2)
    print(time.perf_counter() - start_time, "seconds assimilation finished")
    
    # Do the phrasing. Apply prose or poetry phrasing depending on the genre.
    if genre_1 == "poetry":
        phrasing_list_1 = phrasing_poetry(assimilated_list_1)
    elif genre_1 == "prose":
        phrasing_list_1 = phrasing_prose(assimilated_list_1)
    if genre_2 =="poetry":
        phrasing_list_2 = phrasing_poetry(assimilated_list_2)
    elif genre_2 == "prose":
        phrasing_list_2 = phrasing_prose(assimilated_list_2)
    print(time.perf_counter() - start_time, "seconds phrasing finished")
    
    # Textmatching
    print(time.perf_counter() - start_time, "seconds textmatching started")
    matches, complura_matches = compare_texts(phrasing_list_1, phrasing_list_2, stop_list)
    print(time.perf_counter() - start_time, "seconds textmatching finished")
    
    # Scissa filter
    matches = scissa(matches)
    print(time.perf_counter() - start_time, "seconds scissa filter finished")

    # HTRG filter
    if htrg_switch == True:
        print(time.perf_counter() - start_time, "seconds HTRG filter started")
        matches = htrg(matches)
        print(time.perf_counter() - start_time, "seconds HTRG filter finished")
    elif htrg_switch == False:
        print(time.perf_counter() - start_time, "seconds HTRG filter skipped")

    # Context similarity filter
    if similarity_switch == True:
        print(time.perf_counter() - start_time, "seconds context similarity filter started")
        matches = similarity_filter(matches)
        print(time.perf_counter() - start_time, "seconds context similarity filter finished")
    elif similarity_switch == False:
        print(time.perf_counter() - start_time, "seconds similarity filter skipped")

    # Combine resuls of filtered matching and complura matching, eliminating duplicates
    for cmatch in complura_matches:
        for match in matches: 
            if cmatch[1] != match[1] and cmatch[3] != match[3]:
                matches.append(cmatch)
                break
    
    # Write results to Excel
    matches = [['Number', 'Passage Text A', 'Text A', 'Passage Text B', 'Text B', 'shared words', "POS A", "POS B", "HTRG", "Context similarity"]] + matches 
    write_to_excel(matches, output_excel)
    print(time.perf_counter() - start_time, "seconds citation analysis finished, results written to Excel")

if __name__ == "__main__":
    main()
