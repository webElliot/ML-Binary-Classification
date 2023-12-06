import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

from model import (BinaryClassifier)
import random
import math
import time

random.seed(time.time())


class MachineLearning:
    numerical_features = "Overall,Attack,Defence,Strength,Hitpoints,Ranged,Prayer,Magic,Cooking,Woodcutting,Fletching,Fishing,Firemaking,Crafting,Smithing,Mining,Herblore,Agility,Thieving,Slayer,Farming,Runecraft,Hunter,Construction,Clue Scrolls (all),Clue Scrolls (beginner),Clue Scrolls (easy),Clue Scrolls (medium),Clue Scrolls (hard),Clue Scrolls (elite),Clue Scrolls (master),LMS - Rank,Rifts closed,Abyssal Sire,Alchemical Hydra,Artio,Barrows Chests,Bryophyta,Calvar'ion,Cerberus,Chambers of Xeric,Chaos Fanatic,Commander Zilyana,Crazy Archaeologist,Dagannoth Prime,Dagannoth Rex,Dagannoth Supreme,Deranged Archaeologist,General Graardor,Giant Mole,Grotesque Guardians,Hespori,Kalphite Queen,King Black Dragon,Kraken,Kree'Arra,K'ril Tsutsaroth,Mimic,Nex,Obor,Phantom Muspah,Sarachnis,Skotizo,Spindel,Tempoross,The Gauntlet,The Corrupted Gauntlet,Thermonuclear Smoke Devil,Tombs of Amascut: Expert Mode,TzTok-Jad,Vet'ion,Vorkath,Wintertodt,Zalcano,Zulrah,Soul Wars Zeal,Bounty Hunter - Hunter,Bounty Hunter - Rogue,Chaos Elemental,Corporeal Beast,Nightmare,Tombs of Amascut,Callisto,Chambers of Xeric: Challenge Mode,Phosani's Nightmare,Theatre of Blood,Theatre of Blood: Hard Mode,TzKal-Zuk,Venenatis,PvP Arena - Rank,Scorpia,Bounty Hunter (Legacy) - Hunter,Bounty Hunter (Legacy) - Rogue".split(
        ",")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.001
    step_size = 5
    gamma = 0.3

    batch_size = 1024

    num_workers = 32 # recommended 1 per core
    csv_path = "players.csv"
    model_path = "model.pt"

    def __init__ (self):
        self.model = BinaryClassifier()
        self.load_model()

        self.optimizer = torch.optim.Adam(self.model.parameters() , lr = self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer , step_size = self.step_size , gamma = self.gamma)

        self.criterion = nn.BCELoss()
        self.load_data()

    def load_model (self):  # try to load the model from file
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print("Successfully loaded model from file.")
        except Exception as e:
            print("Failed to load model from file.")
        self.model = self.model.to(self.device)  # move to GPU

    def load_data (self):
        df = pd.read_csv(self.csv_path)
        names = df[ 'Name' ].values

        # Split the data into features and target
        X_df = df.drop(columns = [ 'x' ])
        y = df[ 'x' ].values

        # Preprocess your data
        categorical_features = [ 'Name' ]  # Specify your categorical column names here

        preprocessor = ColumnTransformer(
            transformers = [('num' , StandardScaler() , self.numerical_features) ,]
            )

        # Apply preprocessing
        X = preprocessor.fit_transform(X_df)

        # Convert the preprocessed data back to DataFrame
        df = pd.DataFrame(X , columns = self.numerical_features)

        # Create datasets
        X_train , X_test , y_train , y_test , names_train , names_test = train_test_split(X , y , names ,
                                                                                          test_size = 0.2 ,
                                                                                          random_state = 42)

        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train) , torch.Tensor(y_train))
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_test) , torch.Tensor(y_test))

        self.train_dataloader = DataLoader(train_dataset , batch_size = self.batch_size , shuffle = True , num_workers = self.num_workers)
        self.test_dataloader = DataLoader(test_dataset , batch_size = self.batch_size , shuffle = True , num_workers = self.num_workers)


class MlFunctions(MachineLearning):
    training_iters = 20

    def train (self):
        for epoch in range(self.training_iters):
            self.model.train()
            running_loss = 0.0
            for inputs , labels in self.train_dataloader:
                inputs , labels = inputs.to(self.device) , labels.to(self.device)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze() , labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(self.train_dataloader):.4f}')
            if epoch % 10 == 0:
                self.test_model()
                print(f"Saving {epoch}.")
                torch.save(self.model.state_dict() , self.model_path)  # save the model state

            self.scheduler.step()  # decay the learning rate
        print('Finished Training')

        # save the model state
        torch.save(self.model.state_dict() , self.model_path)

    def test_model (self):
        test_loss , test_accuracy , recall , precision , f1_score = self.evaluate_model()
        print("Test Loss: {:.4f}, Test Accuracy: {:.2f}%, Recall: {:.2f}, Precision: {:.2f}, F1-Score: {:.2f}".format(
            test_loss , test_accuracy , recall , precision , f1_score))

    def evaluate_model (self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_labels = [ ]
        all_predictions = [ ]

        with torch.no_grad():
            for inputs , labels in self.test_dataloader:
                inputs , labels = inputs.to(self.device) , labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze() , labels)

                total_loss += loss.item()

                # Calculate accuracy
                predicted_labels = (outputs >= 0.5).float().squeeze()  # Thresholding at 0.5 for binary classification
                correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

                # Save the labels and predictions for later use
                all_labels.extend(labels.detach().cpu().numpy())
                all_predictions.extend(predicted_labels.detach().cpu().numpy())

        average_loss = total_loss / len(self.test_dataloader)
        accuracy = (correct_predictions / total_samples) * 100.0

        # Calculate recall, precision, F1 score
        classification_metrics = classification_report(all_labels , all_predictions , output_dict = True)
        recall = classification_metrics[ 'weighted avg' ][ 'recall' ]
        precision = classification_metrics[ 'weighted avg' ][ 'precision' ]
        f1_score = classification_metrics[ 'weighted avg' ][ 'f1-score' ]

        print("Total Correct Predictions:" , correct_predictions)
        print("Total Samples:" , total_samples)

        return average_loss , accuracy , recall , precision , f1_score


if __name__ == "__main__":
    mlHandle = MlFunctions()
    mlHandle.train()
    mlHandle.test_model()