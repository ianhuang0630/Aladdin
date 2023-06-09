import sys
import os

from sem_complete.sem_complete.Brainstorm import IterativeBrainstormer
from sem_complete.sem_complete.Dreamstorm import Dreamstormer

if __name__ == '__main__':

    dream_text = "I am in a forest with mom and we’re arguing. We’re lost, and I’m trying to find our way. She keeps telling me that we’re circling the same place over and over again. But this forest is no ordinary forest. There are creatures here that I have never seen before, and creatures that talk. We stop in front of an ancient tree, which hosts a gaping hole. In the hole sits an owl, who begins to speak. For some reason, I begin to argue with the owl. It glares at me and threatens me by slashing its wing against its throat. I run."
#     dream_text = """
# I'm in adult form and there I am going to band class, and then a concert. It's rush time people are rushing to go to "our" band concert, my bandmates. I think I play the clarinet just like I did in real life in junior high. I don't see the intsrument much. I go into a building I don't know how I got in that isn't shown in the dream. I find the girls dressing room and there are other girls in there getting dressed, but it's very blurry I just see figures moving fast. I'm the one that's late because there's a band concert tonight and we all have to play in it. While in the dressing room I wanted to do my hair and put on my outfit. Then I went somewhere it was like a pause in the dream. I then get this feeling and see everyone is GONE and dressed they're already in the "ballroom" getting ready to play for the audience. So when I noticed people are gone and I'm the one that's late the dream turns into a maze. I try to find my way to the concert room where everyone else is, and have a hard time finding, and then I do. It then begins, I made it and I'm there playing a little bit with the rest of the pack. I also felt like I didn't know how to play my instrument for a moment, and had anxiety about that, I played on anyway. The concert was short so I oddly don't see our band playing much.
# """
    output = Dreamstormer().run(dream_text)

    import ipdb; ipdb.set_trace() 


