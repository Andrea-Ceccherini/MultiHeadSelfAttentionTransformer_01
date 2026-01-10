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

Command                                                                                                 Purpose
git remote add origin https://github.com/Andrea-Ceccherini/MultiHeadSelfAttentionTransformer_01.git     Links your local repository to the GitHub one. Replace <REMOTE_URL> with the URL you copied from GitHub (e.g., https://github.com/Andrea-Ceccherini/MultiHeadSelfAttentionTransformer_01.git . origin is the default name for the remote.
git branch -M main                                                                                      Renames your current local branch to main. (In older Git versions, this might be master, the modern standard name is main).


C. Push Code
Finally, push your local commits to the remote repository on GitHub.

Command                             Purpose
git push -u origin main             Pushes your changes (commits) from the local main branch to the origin remote. The -u flag sets the upstream, so future pushes from this branch can be simply git push and git pull.


You may be prompted to enter your GitHub credentials or use a Personal Access Token (PAT) for authentication.

After this, refresh your GitHub repository page, and you should see all your project files.


================================================================


================================================================
Phase 1
Phase 1 creates just the "Brain" (.safetensors)
================================================================

================================================================
Phase 2
Phase 2 creates the "Brain," the "Skeleton" (Config), and the "Dictionary" (Tokenizer).
When Phase 2 finishes, you will find a folder named supervised_qa_model_files containing roughly 5 to 7 files:
1. The Weights (The Brain)

    File: fine_tuned_best.safetensors (approx 1.3 GB)

    What is it? This is exactly like your Phase 1 file, but updated with the Liver knowledge.

2. The Configuration (The Skeleton)

    File: config.json

    What is it? A small text file that tells any program: "This model has 12 layers, 768 dimensions, and 12 heads."

    Why? This allows you to load the model in the future without manually typing CustomTransformer(num_layers=12...). You can just say AutoModel.from_pretrained().

3. The Tokenizer (The Dictionary)

    Files:

        vocab.json

        merges.txt

        tokenizer.json

        tokenizer_config.json

    What are they? These are the files that convert text ("Liver") into numbers ("1542").

    Why? By saving these inside the model folder, you ensure that the model always travels with its specific dictionary. You won't need to load the separate load_gpt2_tokenizer() function anymore.
==============================================================

==============================================================
llama.cpp ENVIRONMENT

1) Check if llama.cpp ENVIRONMENT does exist in Ubuntu.
Enter the command
    ls -d ~/llama.cpp
If th answer is:
    ls: cannot access '/home/andrea/llama.cpp': No such file or directory
means that llama.cpp is not installed
2) Install llama.cpp environment in Ubuntu
Go to your home folder
cd ~

# 2.1 Download the source code
git clone https://github.com/ggerganov/llama.cpp

# 2.2. Enter the new folder
cd llama.cpp

# 2.3. install cmake on your Ubuntu system.
sudo apt update
sudo apt install cmake build-essential
sudo apt install libcurl4-openssl-dev

# 2.4 Run these commands inside your ~/llama.cpp folder:
mkdir build
cd build

# 2.5 Inside the build folder
cmake ..
cmake --build . --config Release -j 4

# 2.6 Inside the virtual environment navigate to 
(.venv) andrea@pc-matteo:~/llama.cpp$

# 2.7 Install requirements
pip install -r requirements.txt


