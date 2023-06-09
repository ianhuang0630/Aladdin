import requests

asd = {'abstract_scene_description': 'a fancy french restaurant',
        'recursion_levels': 1, # useless
         }

semcomplete_out = requests.post('http://localhost:5000/semcomplete/', json=asd)

print(semcomplete_out.json())

objr = {
    'asset_property_list': semcomplete_out.json(),
    'topk': 3
}
retrieval_out = requests.post('http://localhost:5000/retrieve/', json=objr)
print(retrieval_out.json())

tex = {'objects': retrieval_out.json(),
'object_index': 0,
'option_index': 0,
'preview': True
}

tex_out = requests.post('http://localhost:5000/stylize/', json=tex)

print('==============================')
print(tex_out.json())

