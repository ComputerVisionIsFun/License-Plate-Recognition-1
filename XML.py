import xml.etree.ElementTree as ET
import os

def xmlDecoding(xmlPath)->dict:
    
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

def xmlDecodingInTheFolder(xmlFolder)->list:
    # print(xmlFolder,'--------------------xmldecodinginthefolder')
    files = os.listdir(xmlFolder)
    # print(files)
    annos = []
    objs = []
    for file in files:
        if file[-1]=='l':
            xmlPath = xmlFolder + file
            anno = xmlDecoding(xmlPath)
            # print(xmlPath, anno)
            annos.append(anno)
            objs+=anno['objs']

    return annos, objs
