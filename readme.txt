1. how to push code just to sainath branch?
    git checkout sainath                --> switch to sainath branch
    git add .                           --> add all the files to the staging area
    git commit -m "Your commit message" --> commit the changes
    git pull origin sainath --rebase    --> pull the code from the remote sainath branch
    git push origin sainath             --> push the code to the remote sainath branch

2. how to push code to the main branch?
    git checkout main --> switch to main branch
    git add .
    git commit -m "Your commit message"
    git push origin main

3. how to pull code from main branch to sainath branch?
    git checkout sainath
    git pull origin main

4. how do I know which branch I am in?
    git branch