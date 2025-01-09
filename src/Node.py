
class Node():
    
    id = 0
    computational_power = 0
    memory = 0
    eventrates = [] 
    
    
    def __init__(self, id:int ,compute_power: int, memory: int):
        self.id = id
        self.computational_power = compute_power
        self.memory = memory
        self.eventrates = []
        self.Parent = []
        self.Child = []
        self.Sibling = []
        
    

    def __str__(self):
        parent_ids = [parent.id for parent in self.Parent] if self.Parent else None
        child_ids = [child.id for child in self.Child] if self.Child else None
        sibling_ids = [sibling.id for sibling in self.Sibling] if self.Sibling else None

        return (f"Node {self.id}\n"
                f"Computational Power: {self.computational_power}\n"
                f"Memory: {self.memory}\n"
                f"Eventrates: {self.eventrates}\n"
                f"Parents: {parent_ids}\n"
                f"Child: {child_ids}\n"
                f"Siblings: {sibling_ids}\n")

        