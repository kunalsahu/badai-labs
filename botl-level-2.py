def respond(sentence):
    """Parse the user's inbound sentence and find candidate terms that make up a best-fit response"""
    cleaned = preprocess_text(sentence)
    parsed = TextBlob(cleaned)
    ​
        # Loop through all the sentences, if more than one. This will help extract the most relevant
        # response text even across multiple sentences (for example if there was no obvious direct noun
        # in one sentence
        
    pronoun, noun, adjective, verb = find_candidate_parts_of_speech(parsed)
    ​
        # If we said something about the bot and used some kind of direct noun, construct the
        # sentence around that, discarding the other candidates
    resp = check_for_comment_about_bot(pronoun, noun, adjective)
    ​
        # If we just greeted the bot, we'll use a return greeting
    if not resp:
            resp = check_for_greeting(parsed)
    ​
    if not resp:
            # If we didn't override the final sentence, try to construct a new one:
            if not pronoun:
                resp = random.choice(NONE_RESPONSES)
            elif pronoun == 'I' and not verb:
                resp = random.choice(COMMENTS_ABOUT_SELF)
            else:
                resp = construct_response(pronoun, noun, verb)
    ​
        # If we got through all that with nothing, use a random response
    if not resp:
            resp = random.choice(NONE_RESPONSES)
            logger.info("Returning phrase '%s'", resp)
        # Check that we're not going to say anything obviously offensive
    filter_response(resp)
    ​
    return resp
    ​
    def find_candidate_parts_of_speech(parsed):
        """Given a parsed input, find the best pronoun, direct noun, adjective, and verb to match their input.
        Returns a tuple of pronoun, noun, adjective, verb any of which may be None if there was no good match"""
        pronoun = None
        noun = None