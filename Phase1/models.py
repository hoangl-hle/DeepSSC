import torch
import torch.nn as nn

class GEautoencoder(nn.Module):
    def __init__(self, fan_in):
        super(GEautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fan_in, 4096),
            nn.BatchNorm1d(4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ELU(),
            nn.Linear(4096, fan_in)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNAautoencoder(nn.Module):
    def __init__(self, fan_in):
        super(CNAautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fan_in, 4096),
            nn.BatchNorm1d(4096),
            nn.ELU(),
            nn.Linear(4096, 1024),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ELU(),
            nn.Linear(4096, fan_in)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Subtyping_model(nn.Module):
    def __init__(self, ge_encoder, cna_encoder, subtypes):
        super(Subtyping_model, self).__init__()
        
        self.ge_repr = nn.Sequential(*list(ge_encoder.children())[1:])
        self.cna_repr = nn.Sequential(*list(cna_encoder.children())[1:])
        
        self.classifier = nn.Sequential(
            nn.Linear(2048+1024, 1024),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )
        
    def forward(self, x1, x2):  
        ge_ft = self.ge_repr(x1)
        cna_ft = self.cna_repr(x2)

        return self.classifier(torch.hstack((ge_ft, cna_ft)))