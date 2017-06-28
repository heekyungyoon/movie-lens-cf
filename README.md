# movie-lens-cf
classic collaborative filtering with movie lens data


## User-based CF vs Item-based CF
Whether it's user-based or item-based, there's no fundamental difference between two methods. If there are approximately same users and items, you can pick either way. 
However, when either user or item grows much bigger than the other, the time required to find neighbors grows. This is when it becomes clear which one to choose.  

Suppose a single user has rated A items on average and top k is 30. (M users and N items in total)  
- User-based: for 30 similar users to a user, examine A itmes per similar user.  
    -> Searching 30 similar users among M users * A items  
- Item-based: a user has rated A items, and we search for maximum 30 similar items for each rated item.  
    -> A items * Searching 30 similar items among N items  

As you can see, the only difference between user-based and item-based cf comes from the time required for searching neighbors.



## Quick Start
```sh
python recommendation.py
```
