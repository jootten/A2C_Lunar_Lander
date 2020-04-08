from coordinator import Coordinator
import ray
def main(): 
    ray.init(
    memory=500 * 1024 * 1024,
    object_store_memory=400 * 1024 * 1024,
    driver_object_store_memory=100 * 1024 * 1024)

    coord = Coordinator()
    coord.run_for_episodes()
    #returns model at the end?
    ray.shutdown()

if __name__ == '__main__':
    main()
    
