import xml.etree.ElementTree as ET
import os


def xmlDecoding(xmlPath)->dict:
    '''
    ## return:annotation = {'filename':str, 'objs':[]} 
    
    '''
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    annotation = {}
    annotation.update({'filename':root.find('filename').text})
    annotation.update({'objs':[]})
    objs = root.findall('object')

    for obj in objs:
        bbox = {'xmin':0,'xmax':0,'ymin':0,'ymax':0}
        for key in bbox.keys():
            bbox[key] = int(float(obj.find('bndbox').find(key).text))
        
        annotation['objs'].append(bbox)

    return annotation

def xmlDecodingInTheFolder(xmlFolder):
    
    files = os.listdir(xmlFolder)
    
    annos = []
    objs = []
    for file in files:
        if file[-1]=='l':
            # xmlPath = xmlFolder + file
            xmlPath = os.path.join(xmlFolder, file)
            anno = xmlDecoding(xmlPath)# anno = {'filenmae': str, 'objs':[{'xmin':, 'xmax', 'ymin':, 'ymax':}, ...]}
            # print(xmlPath, anno)
            annos.append(anno)
            objs+=anno['objs']

    return annos, objs
