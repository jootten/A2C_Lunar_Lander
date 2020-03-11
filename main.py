from coordinator import Coordinator
import ray
def main(): 
    ray.init()
    coord = Coordinator()
    coord.run_for_episodes()
    #returns model at the end?
    breakpoint()
    ray.shutdown()

if __name__ == '__main__':
    main()
    
