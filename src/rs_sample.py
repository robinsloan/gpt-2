#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import re
from wordfilter import Wordfilter
import glob
import json
import random

import model, sample, encoder

wordfilter = Wordfilter()

def filter_and_truncate(text, num_sentences_desired):

    print("number of sentences desired: ", num_sentences_desired)

    fragments = text.split(".")
    num_sentences_added = 0
    new_text = ""

    while ( (num_sentences_added < num_sentences_desired) and (num_sentences_added < len(fragments)) ):
        new_text += fragments[num_sentences_added] + "."
        num_sentences_added += 1

    new_text = re.sub(' +', ' ', new_text)
    char_diff = len(text) - len(new_text)
    print("-"*80)
    print(text)
    print("--BECAME--")
    print(new_text)
    print("FYI, " + str(char_diff) + " chars 'wasted'")
    print("-"*80)
    return new_text

def needs_redo(text):
    rv = False
    if text.find("\"") > 0: # if there is a quotation mark...
        print("Text candidate has a quotation mark; redoing.")
        rv = True
    for fragment in text.split("."):
        if len(fragment.split(",")) > 4: # if there are too many commas...
            print("A part of the text candidate has too many commas; redoing.")
            rv = True
    if wordfilter.blacklisted(text):
        print("Text candidate is blacklisted; redoing.")
        print("For reference, the text was: ", text)
        rv = True
    else:
        print("Text candidate passes.")
    return rv

def generate_story(
    model_name='fantasy1',
    prompts=['In the beginning, there was'],
    seed=None,
    nsamples=1,
    batch_size=1,
    length=80, # this is BPE thingies, not chars! So it goes a lot further
    temperature=1.0,
    top_k=80,
    top_p=0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        # here we come to story generation logic

        for i in range(len(prompts)):
            prompts[i] = prompts[i].split("|") # [0] is the text, [1] is the number of desired sentences
            if len(prompts[i]) > 1:
                prompts[i][1] = int(prompts[i][1])
                if prompts[i][1] < 1:
                    prompts[i][1] = 1
            else:
                prompts[i][1] = 1

        """
        prompts = [[f'The city of {cities[2]} is known for', 2],
                   [f'The most famous resident of {cities[2]} is', 1],
                   [f'The road from {cities[2]} to {cities[1]} is', 2],
                   ["One day,", 3],
                   ["Finally,", 1],
                   [f'When they reached {cities[1]},', 1]]


        prompts = [[f'On the road to {cities[2]}, Nordric and Ludwig encountered a', 2],
                   ["They decided to", 2],
                   ["At last,", 2],
                   [f'When they reached {cities[1]},',2]]
        """

        prompt_index = 0
        running_text_for_output = ""
        running_text_for_context = ""
        run_redo = False

        while (prompt_index < len(prompts)):

            prompt_text = prompts[prompt_index][0]
            num_sentences_desired = prompts[prompt_index][1]

            if run_redo:
                run_redo = False
            else:
                running_text_for_context += " " + prompt_text
                running_text_for_output += " " + prompt_text.upper()

                running_text_for_context = re.sub(' +', ' ', running_text_for_context)
                running_text_for_output = re.sub(' +', ' ', running_text_for_output)

                prompt_index += 1

            context_tokens = enc.encode(running_text_for_context)
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    new_text = enc.decode(out[i])
                    new_text = filter_and_truncate(new_text, num_sentences_desired)
                    if needs_redo(new_text):
                        run_redo = True
                    else:
                        running_text_for_context += new_text + "\n\n"
                        running_text_for_output += new_text + "\n\n"

        print(running_text_for_output)
        return running_text_for_output

def get_json_files(model_name='fantasy1'):

    print("Loading " + model_name)
    prompt_filenames = glob.glob("/home/robin/clash-share/fantasy-data/survey-responses/prompts/*-prompts.txt")
    for prompt_filename in prompt_filenames:
        print("Generating story for prompt file ", prompt_filename)

        prompts = []
        with open(prompt_filename) as prompt_file:
            for cnt, line in enumerate(prompt_file):
                prompts.append(line)

            generated_story = generate_story(model_name, prompts)

            new_filename = prompt_filename.replace(".txt", "_OUTPUT.txt")
            with open(new_filename, "w") as text_file:
                text_file.write(generated_story)

if __name__ == '__main__':
    fire.Fire(get_json_files)