import hashlib
import os

# Constants
salt = "0000000000000000000fa3b65e43e4240d71762a5bf397d5304b2596d116859c"
e = 2 ** 52
client_seed = "0000000000000538200a48202ca6340e983646ca088c7618ae82d68e0c76ef5a"
initial_server_seed = '4252e9815c01082712a678134c69b752558805d5b726c754c19424c9c3ae6684'
n=100
seed_index=n-1


server_seeds = [initial_server_seed]

for i in range(1,n):
    server_seeds.append(hashlib.sha256(server_seeds[i - 1].encode()).hexdigest())


# Generate Game Hash
def generate_game_hash(server_seed, client_seed):
    hmac = hashlib.new('sha256', server_seed.encode())
    hmac.update(client_seed.encode())
    hmac.update(salt.encode())
    return hmac.hexdigest()

# Calculate Game Result
def calculate_game_result(game_hash):
    if int(game_hash, 16) % 33 == 0:
        return 1
    h = int(game_hash[:13], 16)
    # return round((100 * e - h) / (e - h) * 100) / 100
    return (((100 * e - h) / (e-h)) // 1) / 100.0

def calculate_previous_game_result(current_server_seed, client_seed):
    previous_server_seed = hashlib.sha256(current_server_seed.encode()).hexdigest()
    previous_game_hash = generate_game_hash(previous_server_seed, client_seed)
    previous_game_result=calculate_game_result(previous_game_hash)
    return [previous_server_seed,previous_game_result]



from fastapi import FastAPI,WebSocket, WebSocketDisconnect,Depends,UploadFile , File,Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_new_game")
async def get_scrapped_content():
    global seed_index  # Declare seed_index as global to modify it
    if seed_index>=0:
        game_hash =generate_game_hash(server_seeds[seed_index], client_seed)
        game_result = calculate_game_result(game_hash)
        seed_index=seed_index-1
        return {'game_result':game_result,'game_hash':game_hash,'server_seed':server_seeds[seed_index+1],'seed_index':seed_index+1}
    else:
        return {'data':'Sorry, we ran out of seeds!'}
        


class CurrentSeedHashData(BaseModel):
    current_server_seed: str
    
    
@app.post("/verify_prev_game")
async def verify_prev_game(current_server_seed: str = Form(...)):
    try:        
        previous_server_seed,previous_game_result=calculate_previous_game_result(current_server_seed,client_seed)    
        return {'previous_game_result':previous_game_result,'previous_server_seed':previous_server_seed}
    except Exception as e:
        print(e)
        return {'error':str(e)}



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastapi.responses import FileResponse


async def create_plot_to_test(server_seeds):
    game_seed_hash= server_seeds[0]
    first_game_seed_hash = server_seeds[-1]

    results = []
    count = 0
    while game_seed_hash != first_game_seed_hash:
        count += 1
        previous_server_seed,previous_game_result=calculate_previous_game_result(game_seed_hash,client_seed)
        results.append(previous_game_result)
        game_seed_hash =previous_server_seed
        
    results = np.array(results)
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})

    plt.hist(results, range=(0, 25))
    plt.title("Histogram of Game Results", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Number of Games", fontsize=15)
    plt.xlabel("Multiplier", fontsize=15)
    
    # Save the plot as an image file
    plt.savefig('histogram_of_game_results.png')

    plt.close()  # Close the plot to free up memory
    
    file_path= "histogram_of_game_results.png"
    return file_path


        


    
@app.post("/test_stretegy")
async def test_stretegy(number_of_server_seeds: int = Form(...)):
    try:        
        global initial_server_seed
        server_seeds = [initial_server_seed]
        for i in range(1,number_of_server_seeds):
            server_seeds.append(hashlib.sha256(server_seeds[i - 1].encode()).hexdigest())
        
        image_path = await create_plot_to_test(server_seeds) #/home/nagesh/Desktop/Aviator API/app/histogram_of_game_results.png
        return FileResponse(image_path, media_type="image/png", filename="histogram_of_game_results.png")
    except Exception as e:
        print(e)
        return {'error':str(e)}
