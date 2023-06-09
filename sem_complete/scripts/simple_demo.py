import sys
import os

from sem_complete.sem_complete.Brainstorm import OnePassBrainStormer, IterativeBrainstormer
import json


if __name__=='__main__':

    # get the  Brainstorm module. ask it a question about what one would find in a 5-year old's bedroom, providing it a template.
    
    
    assert len(sys.argv) >= 3

    brainstorm_type = sys.argv[1]    

    if brainstorm_type == "iterative":
        description = sys.argv[2]
        brainstormer = IterativeBrainstormer()
        output = brainstormer.run(description, 1)

    elif brainstorm_type == "onepass":
        
        description = sys.argv[2]
        save_dir = sys.argv[3]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Brainstorming...")
        
        load_previous = True
        if load_previous: 
            with open(os.path.join(save_dir, "interm_brainstorm.json"), "r") as f:
                output = json.load(f)

        else:
            brainstormer = OnePassBrainStormer()
            output = brainstormer.run(description=description.lower())
            with open(os.path.join(save_dir, "interm_brainstorm.json"), "w") as f:
                json.dump(output, f)
        print("Done.")     

        print('=======================================') 
        print(output)
    # load template. 

    # print('=======================================') 
    # print("Retrieving images...")
    # with open('credentials/googledev_key', 'r')  as f:
    #     devkey = f.readlines()
    #     devkey = devkey[0].strip()
    # with open('credentials/googlecx_id', 'r')  as f:
    #     cx = f.readlines()
    #     cx = cx[0].strip()
    # gis = GImageSearch(devkey, cx)

    # for asset in output['category_attributes']:
    #     attributes = output['category_attributes'][asset]
    #     search_query = 'photo of ' + ' '.join(attributes) + ' ' + asset
    #     print(search_query)
    #     gis.search(search_query, os.path.join(save_dir, "asset_pictures", asset))
     
