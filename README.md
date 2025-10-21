============================================================================
GIT

Creating a GitHub repository and pushing your local project to it involves two main steps: setting up the repository on GitHub and then using Git commands in your local project folder to link and push your code.

1. Create a Repository on GitHub

1.1 Go to GitHub and log in to your account.
1.2 In the upper-right corner, click the + sign, then select New repository.
1.3 Fill in the details:
    Repository name: Choose a short, memorable name (e.g., my-local-project).
    Description (optional): Add a brief description.
    Visibility: Choose Public or Private.
1.4 Important: For an existing local project, do not check "Initialize this repository with a README" or add a .gitignore or license file.
1.5 You want the remote repository to be empty.
1.6 Click Create repository.
Once created, GitHub will show a "Quick setup" page with instructions. Look for the section titled "…or push an existing repository from the command line" and copy the repository URL (either HTTPS or SSH)



2. Push Your Local Repository to GitHub
Open your terminal or command prompt, navigate to your local project's root directory (the folder containing all your code files), and run the following commands in order.

A. Initialize and Commit Local Project. Only if your local project is not already a Git repository otherwise skip A

Command                             Purpose
git init                            Initializes a new local Git repository.
git add .                           Adds all files in the current directory to the staging area.
git commit -m "Initial commit"      Creates a snapshot (commit) of the staged files with a descriptive message.

B. Link Local to GitHub Repository
Now, connect your local repository to the empty one you created on GitHub.

Command                             Purpose
git remote add origin <REMOTE_URL>  Links your local repository to the GitHub one. Replace <REMOTE_URL> with the URL you copied from GitHub (e.g., https://github.com/user/repo-name.git). origin is the default name for the remote.
git branch -M main                  Renames your current local branch to main. (In older Git versions, this might be master, the modern standard name is main).

C. Push Code
Finally, push your local commits to the remote repository on GitHub.

Command                             Purpose
git push -u origin main             Pushes your changes (commits) from the local main branch to the origin remote. The -u flag sets the upstream, so future pushes from this branch can be simply git push and git pull.


You may be prompted to enter your GitHub credentials or use a Personal Access Token (PAT) for authentication.

After this, refresh your GitHub repository page, and you should see all your project files.


================================================================



