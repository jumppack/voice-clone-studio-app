"""The six guided recording steps. 36-42 words each (~15 s), common words, varied phonemes,
no brand or real-person names. Word counts are checked by tests/test_backend.py."""

STEPS = [
    {"id": 1, "tone": "Warm and neutral", "label": "Explaining",
     "hint": "Talk like you're explaining something to a friend. Relaxed, clear, not rushed.",
     "script": "Okay, so here's the thing about learning anything new: it feels slow at first, then suddenly "
               "it clicks. You just keep showing up, stay curious, and trust that the small steps are adding up "
               "to something bigger."},
    {"id": 2, "tone": "High energy", "label": "Hype",
     "hint": "Big energy! Smile while you talk, push the volume up a little, but don't shout into the mic.",
     "script": "Oh, that was incredible! Did you see that? I can't believe it actually worked! Let's go, let's go! "
               "That is exactly how you do it, and I am not even going to pretend I'm calm right now!"},
    {"id": 3, "tone": "Calm and deadpan", "label": "Deadpan",
     "hint": "Flat and steady, almost bored. Keep your pitch level and your pace even.",
     "script": "Right. So the printer jammed again, the coffee machine is empty, and the meeting moved to seven. "
               "Which is fine. Everything is fine. I will simply sit here, drink some water, and wait for the "
               "weekend to arrive."},
    {"id": 4, "tone": "Curious", "label": "Question",
     "hint": "Genuinely wondering. Let your voice rise at the end of each question.",
     "script": "Wait, have you ever wondered why the sky turns orange in the evening? Or how birds know which way "
               "to fly in winter? What would happen if we could hear colors? Isn't it strange how much we still "
               "don't know?"},
    {"id": 5, "tone": "Excited reaction", "label": "Reaction",
     "hint": "You just got amazing news. Surprised, delighted, a little breathless.",
     "script": "No way, are you serious? You found it under the couch the whole time? That is the best news I've "
               "heard all week! Honestly, I was ready to give up, but look at this, it's perfect, just perfect!"},
    {"id": 6, "tone": "Storytelling", "label": "Story",
     "hint": "Like reading a bedtime story. Take your time, let the pauses breathe.",
     "script": "Long ago, in a quiet village by the sea, there lived a girl who collected lost things. Every morning "
               "she walked along the shore, picking up shells, old keys, and broken clocks, wondering who had once "
               "loved them."},
]
MIN_TAKES = 3
