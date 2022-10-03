'''
Formula of a neural network:
Z = W0 + W1X1 + W2X2 + … + WnXn
Here Z is the output, W0 is the bias, W1, W2, …, Wn are the weights, and X1, X2, …, Xn are the inputs.

epoch is the number of times the entire training data is passed forward and backward through the neural network.

THE val VARIABLE BELOW CONTAINS THE LIST OF DATASET THAT IS TO BE USED FOR NEURAL NETWORK TRAINING AND TESTING
THE val VARIABLE IS A LIST OF TUPLES, WHERE EACH TUPLE CONTAINS THE INPUTS, OUTPUTS AND TRUE VALUE OF ANY LOGICAL OPERATOR

'''

val=[(0,0,1),(0,1,0),(1,0,1),(1,1,0)] 
'''
Here is the dataset for login XOR operator, you can change the dataset to any other logical operator of your choice, 
if that dosent convince you, then you can also create your own binary dataset with 2 outputs and 1 input and make the neural network learn it
'''

learningRate=float(input("Enter learning rate: ")) # learning rate determines how fast the neural network learns
print("The weights value between 0 to 1 is preferred")
w1=float(input("Enter weight 1: "))
w2=float(input("Enter weight 2: "))
w3=float(input("Enter weight 3: "))

print("Lets begin training the neural network")
print('x1\t x2\t trueValue\t output\t \u0394w1\t w1\t \u0394w2\t w2\t \u0394w3\t w3')
done=0
epoch = 0
while (done!=len(val)): # while loop is used to repeat the calculation until all the samples are correctly classified
    done=0
    print(f'\n-----------------------------------epoch {epoch}-------------------------------------------')
    epoch += 1
    for item in val:
        x1,x2,trueValue = item

        outputValue = int(((x1*w1)+(x2*w2)+(1*w3))>0) # Calculating the Output of the neuron using the weights and inputs
        # The activation function is used to convert the output of the neural network into a value between 0 and 1.


        # Calculating errors and finding delta weights
        # delta weights are used to update the weights of the neural network, it describes how far the output is from the true value
        dw1=learningRate*x1*(trueValue-outputValue)
        dw2=learningRate*x2*(trueValue-outputValue)
        dw3=learningRate*1*(trueValue-outputValue)

        # Updating the weights
        w1=round(w1+dw1, 2) #round the values upto 2 decimal places
        w2=round(w2+dw2, 2)
        w3=round(w3+dw3, 2)

        if dw1==0 and dw2==0 and dw3==0: # Training is complete when the delta weights are equal to zero
            done+=1 # done variable used to end the while loop when all the delta weights are zero in a epoch

        print(f'{x1}\t {x2}\t {trueValue}\t {outputValue}\t {dw1}\t {w1}\t {dw2}\t {w2}\t {dw3}\t {w3}')

print("Training complete")
print()
print(f'\nInitial weight w1: {w1}\nInitial weight w2: {w2}\nInitial weight w3: {w3}')
print()
print(f"Therefore the neural network equation for the given dataset is:  Z = {w1}*x1 + {w2}*x2 + {w3}")
print("The output of the neural network is: 0 if Z<=0 else 1")
