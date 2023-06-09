from typing import Union

class SceneQueryTemplate(object):
    """ SceneQueryTemplate
    """
    def __init__(self, tmplt_src: Union[str, list]):
        """
        Args:
            tmplt_src: path to template file
        """
        if isinstance(tmplt_src, str):
            with open(tmplt_src, 'r') as f:
                template_text = f.readlines()
        
        if isinstance(tmplt_src, list):
            template_text = []
            for t in tmplt_src:
                with open(t, 'r') as f:
                    template_text.extend(f.readlines())
                    template_text.append("\n")
        
        self.template_text = template_text
        self.separator = "==============================\n"

    def apply(self, application_template: str , scene_description: str):
        """ Returns the query string
        Args:
            scene_description: a phrase that vaguely describes the desired scene
        Return:
            prompt: a prompt to pass into GPT3.
        """
        assert "{}" in application_template
        out = application_template.format(scene_description) + '\n'
        
        prompt = ''.join(self.template_text + [self.separator, out])
        return prompt, ''.join(self.template_text + [self.separator]), out
    
    def apply_directly(self, question):
        out = question + '\n'
        prompt = ''.join(self.template_text + ["\n"] + [self.separator, out])
        return prompt, ''.join(self.template_text + ["\n"] + [self.separator]), out


if __name__ == '__main__':
    sqt = SceneQueryTemplate('sem_complete/templates/frenchrestaurant_nodelist.txt')
    prompt = sqt.apply('Here is a list of items one would find in {}', 'a messy living room:')    
    print(prompt)