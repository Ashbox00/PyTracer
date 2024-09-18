import numpy as np
import matplotlib.pyplot as plt

def reflected(vector,axis):
    return vector - 2 * np.dot(vector,axis) * axis
def normalize (vector):
    return vector / np.linalg.norm(vector)


def sphere_intersect(center,radius,ray_origin,ray_direction):
    b = 2 * np.dot(ray_direction,ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) /2 
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1,t2)
        return None

# find the nearest object that a ray intersects, if it exists 
#loops through the sphere, searchs for intersections and keep the nearest sphere
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'],obj['radius'],ray_origin,ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance
#when calling this function if nearest object is none there is no object intersected by the ray


#our spheres
objects = [
    {'center': np.array([-0.2,0,-1]), 'radius': 0.7, 'ambient': np.array([0.1, 0 , 0]), 'diffuse': np.array([0.7,0,0]), 'specular': np.array([1,1,1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([0.1,-0.3,0]), 'radius': 0.1, 'ambient': np.array([0.149, 0.42 , 0.153]), 'diffuse': np.array([0.173,0.541,0.18]), 'specular': np.array([1,1,1]), 'shininess': 100,'reflection': .5},
    {'center': np.array([-0.9,1.3,-2]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7,0,0.7]), 'specular': np.array([1,1,1]), 'shininess': 100,'reflection': .5},
    {'center': np.array([1.4,0.7,-3]), 'radius': 0.4, 'ambient': np.array([0, 0.1 , 0.1]), 'diffuse': np.array([0,0.7,0.7]), 'specular': np.array([1,1,1]), 'shininess': 100,'reflection': .5},
    {'center': np.array([-1.2,-0.6,-.9]), 'radius': 0.15, 'ambient': np.array([0.1, 0.1 , 0]), 'diffuse': np.array([0.7,0.7,0]), 'specular': np.array([1,1,1]), 'shininess': 100,'reflection': .5},
    {'center': np.array([0,-9000,0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1 , 0.1]), 'diffuse': np.array([0.6,0.6,0.6]), 'specular': np.array([1,1,1]), 'shininess': 100,'reflection': 1},
]
light = {'position': np.array([-1.3,3.5,2]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1,1,1]), 'specular': np.array([1,1,1])}

light2 = {'center': np.array([-5,10,5]), 'radius': 0.7, 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1,1,1]), 'specular': np.array([1,1,1])}



width, height = 400, 200

max_depth = 4
camera = np.array([0,0,1])
ratio = float(width) / height
screen = (-1, 1/ratio, 1, -1 /ratio) #left, top, right, bottom

image = np.zeros((height,width,3))

for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0],screen[2], width)):  #this block of code is the associate a black color to p
        pixel = np.array([x,y,0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1.1

        for k in range(max_depth):
            nearest_object, min_distance = nearest_intersected_object(objects,origin,direction)
            if nearest_object is None:
                break

            #check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects,origin,direction)
            if nearest_object is None:
                continue

            #compute intersection point between ray and nearest object
            intersection = origin + min_distance * direction



            #checking for an object that is shadowing the light on the object
            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _,min_distance = nearest_intersected_object(objects,shifted_point,intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(   - intersection)
            is_shadowed = min_distance < intersection_to_light_distance


            #RGB
            illumination = np.zeros((3))

        

            

            #ambient
            illumination += nearest_object['ambient'] * light['ambient']
            
            #diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

            #specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

            #reflection 
            color += reflection * illumination
            reflection *= nearest_object['reflection']

            #new ray origin
            origin = shifted_point
            direction = reflected(direction,normal_to_surface)


        image[i,j] = np.clip(color, 0 ,1)







    print("progress: %d/%d" % (i + 1, height))
                          
#ray intersection 
#if the ray (line) that starts at the camrea and goes towards p intersections any object of the scene then:

plt.imsave('imagen.png', image)