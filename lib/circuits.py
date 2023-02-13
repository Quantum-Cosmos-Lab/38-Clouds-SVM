import pennylane as qml
from pennylane import numpy as qnp

n_wires = 4
dev = qml.device('default.qubit', wires=n_wires)

wires = range(n_wires)

#####################################################
@qml.qnode(dev)
def S_state(x):
    wires = range(len(x))
    qml.broadcast(qml.Hadamard, wires, pattern='single')
    qml.broadcast(qml.RZ, wires, pattern='single', parameters=1*qnp.pi*x)
    #qml.broadcast(qml.Hadamard, wires, pattern='single')
    return qml.state()

def S(x):
    """Data-encoding circuit block."""
    qml.broadcast(qml.Hadamard, wires, pattern='single')
    qml.broadcast(qml.RZ, wires, pattern='single', parameters=1*qnp.pi*x)

def W(thetas):
    """Trainable circuit block."""
    qml.broadcast(qml.Rot, wires, pattern='single', parameters = thetas)


@qml.qnode(dev)
def rotations_arbitrary_WS_1(x, thetas):
    scaling = 1
    W(thetas)
    S(scaling*x)
    return qml.state()

@qml.qnode(dev)
def WSWS(x, thetas):
    thetas1 = thetas[:n_wires,:]
    thetas2 = thetas[n_wires:,:]
    scaling = 0.5
    W(thetas1)
    S(scaling*x)
    W(thetas2)
    S(scaling*x)
    return qml.state()

@qml.qnode(dev)
def SELS_1(x, thetas):
    scaling = 1
    qml.StronglyEntanglingLayers(weights=thetas, wires=wires)
    S(scaling*x)
    return qml.state()

@qml.qnode(dev)
def SELSELS_1(x, thetas):
    scaling = 1
    qml.StronglyEntanglingLayers(weights=thetas, wires=wires)
    S(scaling*x)
    return qml.state()

@qml.qnode(dev)
def SELSSELS(x, thetas):
    scaling = 1
    t0 = thetas[0].reshape(1,n_wires,3)
    t1 = thetas[1].reshape(1,n_wires,3)
    qml.StronglyEntanglingLayers(weights=t0, wires=wires)
    S(scaling*x)
    qml.StronglyEntanglingLayers(weights=t1, wires=wires)
    S(scaling*x)
    return qml.state()