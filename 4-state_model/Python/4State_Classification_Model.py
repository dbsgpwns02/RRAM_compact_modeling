import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# --- 시드 고정 ---
seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

plt.close("all")


# --- helper function 정의 ---
def swCalc_IW(i,w):
    swii=np.ones((i.shape[0],))
    swww=np.ones((w.shape[0],))
    for ii in range(i.shape[0]):
        if np.absolute(i[ii])>8e-6:
             swii[ii]=2

        if np.absolute(i[ii])>1.75e-4:
             swii[ii]=36
        if np.absolute(i[ii])>7e-4:
             swii[ii]=138
        #sww[ii]=1/np.absolute(a[ii])
    for ii in range(w.shape[0]-1):
        kk=38

        if w[ii] != w[ii+1]:
             swww[ii]=138
             swww[ii-1]=kk
             swww[ii-2]=kk
             swww[ii+1]=kk
             swww[ii+2]=kk
    return swww.reshape((-1,)),swii.reshape((-1,))


def logtrans(a):
    for ii in range(a.shape[0]):
        if a[ii]>=0:
            a[ii]=np.log10(a[ii])
        else:
            a[ii]=np.log10(-a[ii])
    return a

def log_inv_trans(v,a):
    for ii in range(a.shape[0]):
        if v[ii]>=0:
            a[ii]=np.power(10,a[ii])
        else:
            a[ii]=-1*np.power(10,a[ii])
    return a

# --- 1. 데이터 로드 및 전처리 (4-STATE) ---

# --- [4-STATE logic 1] V3 학습 데이터 로드 ---
try:
    df = pd.read_csv('classified_physics_v3_data.csv')
    print("학습용(Train) 데이터 'classified_physics_v3_data.csv' 로드 성공.")
except FileNotFoundError:
    print("오류: 'classified_physics_v3_data.csv' 파일을 찾을 수 없음...")
    raise SystemExit()
# --- (수정 끝) ---

# --- [4-STATE logic 2] iloc 대신 컬럼명으로 데이터 추출 ---
# (V, I, W 순서가 원본과 다름)
v_train = np.array(df['Voltage(V)']).reshape(-1,1)
y_train = np.array(df['Current(A)']).reshape(-1,1)
w_train = np.array(df['W_physics_state_v3']).reshape(-1,1).astype(float) # (0, 1, 2, 3)

# w_train_0 (t-1 시점의 state) 생성

w_train_0 = np.concatenate((w_train[:1], w_train[:-1]), axis=0).reshape(-1,1).astype(float)
print(f"Train data shape (V): {v_train.shape}")
print(f"Train data shape (W): {w_train.shape}")


# --- [4-STATE logic 3] 2-State용 SPICE supplement 로직 제거 ---
# (V3 데이터는 이미 전이 상태(1, 3)를 포함하므로 이 로직은 불필요하며,
#  '1-w_train_0' (0->1, 1->0) 로직이 4-State에 맞지 않음)
print("Skipping 2-State 'SPICE supplement' logic.")
# for ind in range(w_train_0.shape[0]):
#     if v_train[ind,0] > 0.8 or v_train[ind,0] < -1.2:

sww,swi=swCalc_IW(y_train,w_train_0)

y_train=logtrans(y_train)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

v_train = scaler1.fit_transform(v_train)
y_train = scaler2.fit_transform(y_train)

# --- [4-STATE logic 4] V3 테스트 데이터 로드 ---
try:
    df1 = pd.read_csv('classified_TEST_physics_v3_data.csv')
    print("테스트용(Test) 데이터 'classified_TEST_physics_v3_data.csv' 로드 성공.")
except FileNotFoundError:
    print("오류: 'classified_TEST_physics_v3_data.csv' 파일을 찾을 수 없음..")
    raise SystemExit()

# --- [4-STATE logic 5] iloc 대신 컬럼명으로 데이터 추출 ---
v_test = np.array(df1['Voltage(V)']).reshape(-1,1)
y_test = np.array(df1['Current(A)']).reshape(-1,1)
w_test = np.array(df1['W_physics_state_v3']).reshape(-1,1).astype(float) # (0, 1, 2, 3)

# w_test_0 (t-1 시점의 state) 생성
w_test_0 = np.concatenate((w_test[:1], w_test[:-1]), axis=0).reshape(-1,1).astype(float)

y_test=logtrans(y_test)

v_test = scaler1.transform(v_test)
y_test = scaler2.transform(y_test)

# --- 2. 모델 빌드 (4-STATE) ---
print("Building 4-State model...")
# classification
v_next = tf.keras.Input(shape=(1,),name="Voltage")
w_init = tf.keras.Input(shape=(1,),name="State") # (0, 1, 2, 3 정수 입력)
w_v_init = tf.keras.layers.Concatenate(axis=1)([v_next, w_init])
w_v_init = tf.keras.layers.Dense(units = 20, activation='relu')(w_v_init)
w_next = tf.keras.layers.Dense(units = 20, activation='relu')(w_v_init)

# --- [4-STATE logic 6] 출력 레이어를 4-State (softmax)로 변경 ---
# (units=2 -> 4, activation='sigmoid' -> 'softmax')
w_next = tf.keras.layers.Dense(units = 4, activation='softmax',name='output_w')(w_next)

w_v_next = tf.keras.layers.Concatenate(axis=1)([v_next, w_next])
w_v_next = tf.keras.layers.Dense(units = 20, activation='relu')(w_v_next)
y_next = tf.keras.layers.Dense(units = 20, activation='relu')(w_v_next)
y_next = tf.keras.layers.Dense(units = 1,name='output_y')(y_next)

binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model = tf.keras.Model(inputs=[v_next,w_init], outputs=[w_next,y_next])

# --- [4-STATE logic 7] Loss 함수 확인 ---
# (SparseCategoricalCrossentropy는 0,1,2,3 정수 라벨과 softmax 출력을
#  자동으로 비교하므로 4-State에서도 '정답'인 Loss 함수임.)
model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', 'mse'],loss_weights=[0.5,0.5],metrics=['accuracy','mse'])

# --- [4-STATE logic 8] EarlyStopping 수정 (과적합 방지) ---
# (monitor='loss' -> 'val_loss', patience 100)
callback =tf.keras.callbacks.EarlyStopping(
    monitor='loss', # (검증 손실을 모니터링) => 원래는 val_loss
    mode='min',
    patience=300000,       # (100 epoch간 개선 없으면 중지)
    verbose=1,
    restore_best_weights=True
)

print("--- Model Summary ---")
model.summary()
print("\nStarting model training...")

# --- 3. 모델 학습 ---
hist = model.fit(x=[v_train,w_train_0], y=[w_train, y_train],
                 sample_weight=[sww,swi],
                 batch_size = 32,
                 epochs = 10000, # (최소 10000 epoch로 테스트!!)
                 shuffle=True,
                 verbose=1,
                 validation_split=0.1, # (과적합 방지를 위한 검증 데이터 10%)
                 callbacks=[callback])

print("Training finished.")
weights = model.get_weights()

# ===================================================================
# [시작]  Verilog-A 변환을 위한 가중치/스케일러 저장 코드 
# ===================================================================
# [추가!] 
print("\n--- [DEBUG] Final Model Summary ---")
model.summary()

print("\n--- 3b. Saving All Model Weights for Verilog-A ---")

# model.layers를 순회하며 모든 Dense 레이어의 가중치(W)와 편향(B)을 저장
for layer in model.layers:

    # if 'dense' in layer.name:
    if isinstance(layer, tf.keras.layers.Dense): 

        weights_list = layer.get_weights()

        if weights_list:
            W = weights_list[0]
            B = weights_list[1]

            w_filename = f"{layer.name}_W.txt"
            b_filename = f"{layer.name}_B.txt"

            np.savetxt(w_filename, W, delimiter=',')
            np.savetxt(b_filename, B, delimiter=',')

            print(f" Saved: {layer.name} (Shape W: {W.shape}, B: {B.shape})")
            print(f"   -> {w_filename}")
            print(f"   -> {b_filename}")
        else:
            print(f" Skipped: {layer.name} (No weights found)")
    else:
        # Input, Concatenate 등 Dense가 아닌 레이어는 건너뛰자
        print(f" Skipping non-Dense layer: {layer.name} (Type: {type(layer).__name__})")


print("\n--- 3c. Saving Scaler Parameters for Verilog-A ---")

# (스케일러 저장)
v_min = scaler1.data_min_[0]
v_max = scaler1.data_max_[0]
i_log_min = scaler2.data_min_[0]
i_log_max = scaler2.data_max_[0]

scaler_params = np.array([
    v_min,
    v_max,
    i_log_min,
    i_log_max
])

scaler_filename = "scaler_V_I_params.txt"
np.savetxt(scaler_filename, scaler_params, delimiter=',',
           header="v_min, v_max, i_log_min, i_log_max", comments='')

print(f" Saved: {scaler_filename}")
print(f"  -> Voltage Min/Max: [{v_min}, {v_max}]")
print(f"  -> Log(Current) Min/Max: [{i_log_min}, {i_log_max}]")

print("\n--- [완료] Verilog-A 변환 준비 완료 ---")
# ===================================================================
# ===================================================================

# --- 4. 예측 및 역정규화 (Training Data) ---
# (이하 로직은 np.argmax(axis=1)을 사용하므로 4-State에서도 자동 호환될 것)
print("Predicting on training data...")
train_predict = model.predict([v_train,w_train_0])
w_train_predict = train_predict[0] # (N, 4) shape의 확률
w_train_predict = np.argmax(w_train_predict,axis=1).reshape(-1,1).astype(float) # (N, 1) shape의 0/1/2/3
y_train_predict = train_predict[1]

v_train = scaler1.inverse_transform(v_train)
y_train = scaler2.inverse_transform(y_train)
y_train=log_inv_trans(v_train,y_train)

y_train_predict = scaler2.inverse_transform(y_train_predict)
y_train_predict=log_inv_trans(v_train,y_train_predict)

# --- 5. 예측 및 역정규화 (Testing Data - Autoregressive) ---
# (이하 로직은 np.argmax(axis=1)을 사용하므로 4-State에서도 자동 호환될 것으로 예상 !)
print("Predicting on testing data (autoregressive)...")
w_test_predict = []
y_test_predict = []

# w_init은 (1, 1) shape이어야 함
w = w_test_0[0].reshape(1, 1) # w의 shape을 (1, 1)로 시작

for v in v_test: # v는 (1,) shape
    v_reshaped = v.reshape(1, 1)

    test_predict = model.predict([v_reshaped, w], verbose=0)

    w_prob_next = test_predict[0] # (1, 4) softmax output, 예시 =>  [[0.1, 0.8, 0.05, 0.05]]
    y_pred = test_predict[1]      # (1, 1) array

    # (HSPICE 방식) argmax로 '다음' 상태(0,1,2,3) 결정
    w_label_next = np.argmax(w_prob_next, axis=1).astype(float)[0] # e.g., 1.0

    # '다음' 입력을 (1, 1) shape으로 준비
    w = np.array([[w_label_next]]) # e.g., [[1.]]

    w_test_predict.append(w_label_next) # (결과 저장: 0.0, 1.0, 2.0 or 3.0)
    y_test_predict.append(y_pred[0])

w_test_predict = np.array(w_test_predict).reshape(-1,1)
y_test_predict = np.array(y_test_predict).reshape(-1,1)

print("Inverse transforming test data...")
v_test = scaler1.inverse_transform(v_test)
y_test = scaler2.inverse_transform(y_test)
y_test=log_inv_trans(v_test, y_test)
y_test_predict = scaler2.inverse_transform(y_test_predict)
y_test_predict=log_inv_trans(v_test, y_test_predict)

# --- 6. 정확도 및 MSE 계산 ---
print("Calculating metrics...")
acc_w_train = accuracy_score(w_train,w_train_predict)
acc_y_train = r2_score(y_train,y_train_predict)
mse_w_train = mean_squared_error(w_train,w_train_predict)
mse_y_train = mean_squared_error(y_train,y_train_predict)
acc_w_test = accuracy_score(w_test,w_test_predict)
acc_y_test = r2_score(y_test,y_test_predict)
mse_w_test = mean_squared_error(w_test,w_test_predict)
mse_y_test = mean_squared_error(y_test,y_test_predict)

print("\n--- 훈련(Train) 데이터 성능 ---")
print(f"  State (W) Accuracy: {acc_w_train * 100:.2f} %")
print(f"  Current (I) R2 Score: {acc_y_train:.4f}")
print("\n--- 테스트(Test) 데이터 성능 ---")
print(f"  State (W) Accuracy: {acc_w_test * 100:.2f} %")
print(f"  Current (I) R2 Score: {acc_y_test:.4f}")

# --- 7. 그래프 출력 ---
print("Generating plots...")

# figure 1
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('current in train data')
plt.plot(y_train,c='b',label='true data')
plt.plot(y_train_predict,c='orange',label='predict data')
plt.xlabel('Time (nsec)')
plt.ylabel('Current (A)')
plt.legend()
plt.show()

# figure 2
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('current in test data')
plt.plot(y_test,c='b',label='true data')
plt.plot(y_test_predict,c='orange',label='predict data')
plt.xlabel('Time (nsec)')
plt.ylabel('Current (A)')
plt.legend()
plt.show()

# ... (이하 모든 그래프 코드는 4-State (0,1,2,3)에서도 정상 동작) ...

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('voltage in train data')
plt.plot(v_train,c='b',label='true data')
plt.xlabel('Time (nsec)')
plt.ylabel('Voltage')
plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('voltage in test data')
plt.plot(v_test,c='b',label='true data')
plt.xlabel('Time (nsec)')
plt.ylabel('Voltage')
plt.show()

# (4-State가 0, 1, 2, 3으로 잘 나오는지 확인)
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('state in train data (4-STATE)')
plt.plot(w_train,c='b',label='true data (V3 Label)')
plt.plot(w_train_predict,c='orange',label='predict data', linestyle='--')
plt.xlabel('Time (nsec)')
plt.ylabel('State (0, 1, 2, 3)')
plt.legend()
plt.show()

# (4-State가 0, 1, 2, 3으로 잘 나오는지 확인)
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('state in test data (4-STATE)')
plt.plot(w_test,c='b',label='true data (V3 Label)')
plt.plot(w_test_predict,c='orange',label='predict data', linestyle='--')
plt.xlabel('Time (nsec)')
plt.ylabel('State (0, 1, 2, 3)')
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('IV in train data')
plt.plot(v_train,y_train,c='b',label='true data')
plt.scatter(v_train,y_train_predict,c='orange',label='predict data',s=25)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.legend()
plt.show()

#수정 해야함 (완료 !)
plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('IV in test data') # <- "Linear" 그래프
plt.plot(v_test,y_test,c='b',label='true data')
plt.scatter(v_test,y_test_predict,c='orange',label='predict data',s=25)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)') # <- Y값이 'Current(A)' 그대로임
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.rcParams["font.size"] = 14
plt.title('IV in test data (Log Scale)')
# (log(0) 방지를 위해 1e-12 추가)
plt.plot(v_test,np.log10(np.absolute(y_test) + 1e-12),'rs-',label='true data')
plt.plot(v_test,np.log10(np.absolute(y_test_predict) + 1e-12),'bo-',label='predict data')
plt.xlabel('Voltage (V)')
plt.ylabel('Log Current (A)')
plt.legend()
plt.show()

# --- 8. 결과 CSV 저장 ---

# --- [4-STATE logic 9] 4-State 결과 파일명 변경 ---
print("Saving 4-State results to CSV...")
df = pd.DataFrame(np.concatenate((v_test,w_test,w_test_predict,y_test,y_test_predict),axis=1))
filepath1= 'test_4STATE_V3_PREDICT.csv'
df.to_csv(filepath1,header=['Voltage_V','State','State_pred','Current_A','Current_pred_A'],index=False)

print(f"\nSuccessfully saved test results to '{filepath1}'")
print("All tasks finished.")