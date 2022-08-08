import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
import random

# torch.manual_seed(1)    # reproducible
# 此版本是用FHRCNN來fit FHRCNN_dataset1 
# 未加入局部更新
# 增加avg loss
# (x,y,w,h)全新架構


def per(m, M) : 
    return M - m

def perj(x , m , M) : 
    return max( M - m , x - m , M - x )

def forward_mj (x, s, m, M) :
    m_output = torch.FloatTensor([-(s**2)*(perj(x , m , M)-per( m, M))**2])
    output = torch.exp(m_output)
    return output



class FHRCNN_Network(object) :
    def __init__(self, k = 31,lr = 0.01, epochs = 200, forward_mj = forward_mj):
        super(FHRCNN_Network, self).__init__()
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.forward_mj = forward_mj

        self.w = torch.randn(k) 
        self.b = torch.randn(1)

        # 模糊集合
        self.middle = torch.arange(13,416,13) # 將416*416影像分成31個模糊集合
        self.epsilon =  2
        self.M = self.middle + self.epsilon
        self.m = self.middle - self.epsilon
        self.s = torch.ones(31)
        
        # model weights
        self.model_M = torch.FloatTensor([])
        self.model_m = torch.FloatTensor([])
        self.model_s = torch.FloatTensor([])
        self.model_w = torch.FloatTensor([])
        self.model_b = torch.FloatTensor([])

    def train(self, x1, y) :
        error = []
        for epoch in range(self.epochs):

            cal_loss = 0
            cal_error = 0

            for i in range(len(x1)):

                mjoutput_x = torch.FloatTensor([])
                mjoutput_y = torch.FloatTensor([])
                mjoutput_w = torch.FloatTensor([])
                mjoutput_h = torch.FloatTensor([])

                for j in range(len(self.middle)):
                #forward pass
                    
                    mjoutput_x_temp =  torch.FloatTensor([forward_mj(x1[i][0], self.s[j] , self.m[j] , self.M[j] )])
                    mjoutput_x = torch.cat((mjoutput_x, mjoutput_x_temp))

                    mjoutput_y_temp =  torch.FloatTensor([forward_mj(x1[i][1], self.s[j] , self.m[j] , self.M[j] )])
                    mjoutput_y = torch.cat((mjoutput_y, mjoutput_y_temp))

                    mjoutput_w_temp =  torch.FloatTensor([forward_mj(x1[i][2], self.s[j] , self.m[j] , self.M[j] )])
                    mjoutput_w = torch.cat((mjoutput_w, mjoutput_w_temp))

                    mjoutput_h_temp =  torch.FloatTensor([forward_mj(x1[i][3], self.s[j] , self.m[j] , self.M[j] )])
                    mjoutput_h = torch.cat((mjoutput_h, mjoutput_h_temp))
                

                final_output_x = torch.dot(mjoutput_x,self.w) + self.b
                final_output_y = torch.dot(mjoutput_y,self.w) + self.b
                final_output_w = torch.dot(mjoutput_w,self.w) + self.b
                final_output_h = torch.dot(mjoutput_h,self.w) + self.b


                print(x1[i],mjoutput_x,mjoutput_y,mjoutput_w,mjoutput_h,
                        self.w,self.b,
                        final_output_x, final_output_y, final_output_w, final_output_h)

                

                loss_x = (y[i][0] - final_output_x).flatten() **2
                print('Loss: {0:.4f}'.format(loss_x[0]))
                loss_y = (y[i][1] - final_output_y).flatten() **2
                print('Loss: {0:.4f}'.format(loss_y[0]))
                loss_w = (y[i][2] - final_output_w).flatten() **2
                print('Loss: {0:.4f}'.format(loss_w[0]))
                loss_h = (y[i][3] - final_output_h).flatten() **2
                print('Loss: {0:.4f}'.format(loss_h[0]))

                cal_loss = cal_loss + ((loss_x[0]+loss_y[0]+loss_w[0]+loss_h[0])/4) #calculate avg loss

                # backward pass 
                error_x = -(y[i][0] - final_output_x).flatten()
                error_y = -(y[i][1] - final_output_y).flatten()
                error_w = -(y[i][2] - final_output_w).flatten()
                error_h = -(y[i][3] - final_output_h).flatten()
                #print('Error: ', error)

                cal_error = cal_error + ((abs(error_x[0])+abs(error_y[0])+abs(error_w[0])+abs(error_h[0]))/4)

                # online update
                self.w = self.w - self.lr * mjoutput_x * error_x
                self.b = self.b - self.lr * error_x
                self.w = self.w - self.lr * mjoutput_y * error_y
                self.b = self.b - self.lr * error_y
                self.w = self.w - self.lr * mjoutput_w * error_w
                self.b = self.b - self.lr * error_w
                self.w = self.w - self.lr * mjoutput_h * error_h
                self.b = self.b - self.lr * error_h


                # X-axis update
                for x_update in range(len(self.middle)):
                    if x1[i][0] > self.M[x_update] : #update M 
                        self.M[x_update] = self.M[x_update] - self.lr * error_x * self.w[x_update] * mjoutput_x[x_update] * (self.s[x_update]**2) * 2 * (perj(x1[i][0], self.m[x_update], self.M[x_update])-per(self.m[x_update], self.M[x_update]))

                    elif x1[i][0] < self.m[x_update] : #update m 
                        self.m[x_update] = self.m[x_update] - self.lr * error_x * self.w[x_update] * mjoutput_x[x_update] * -(self.s[x_update]**2) * 2 * (perj(x1[i][0], self.m[x_update], self.M[x_update])-per(self.m[x_update], self.M[x_update]))
                    
                    self.s[x_update] = self.s[x_update] - self.lr * error_x * self.w[x_update] * mjoutput_x[x_update] * (-2) * self.s[x_update] * (perj(x1[i][0], self.m[x_update], self.M[x_update])-per(self.m[x_update], self.M[x_update]))**2


                # Y-axis update
                for y_update in range(len(self.middle)):
                    if x1[i][1] > self.M[y_update] : #update M 
                        self.M[y_update] = self.M[y_update] - self.lr * error_y * self.w[y_update] * mjoutput_y[y_update] * (self.s[y_update]**2) * 2 * (perj(x1[i][1], self.m[y_update], self.M[y_update])-per(self.m[y_update], self.M[y_update]))

                    elif x1[i][1] < self.m[y_update] : #update m 
                        self.m[y_update] = self.m[y_update] - self.lr * error_y * self.w[y_update] * mjoutput_y[y_update] * -(self.s[y_update]**2) * 2 * (perj(x1[i][1], self.m[y_update], self.M[y_update])-per(self.m[y_update], self.M[y_update]))

                    self.s[y_update] = self.s[y_update] - self.lr * error_y * self.w[y_update] * mjoutput_y[y_update] * (-2) * self.s[y_update] * (perj(x1[i][1], self.m[y_update], self.M[y_update])-per(self.m[y_update], self.M[y_update]))**2
                
                # W-axis update
                for w_update in range(len(self.middle)):
                    if x1[i][2] > self.M[w_update] : #update M 
                        self.M[w_update] = self.M[w_update] - self.lr * error_w * self.w[w_update] * mjoutput_w[w_update] * (self.s[w_update]**2) * 2 * (perj(x1[i][2], self.m[w_update], self.M[w_update])-per(self.m[w_update], self.M[w_update]))

                    elif x1[i][2] < self.m[w_update] : #update m 
                        self.m[w_update] = self.m[w_update] - self.lr * error_w * self.w[w_update] * mjoutput_w[w_update] * -(self.s[w_update]**2) * 2 * (perj(x1[i][2], self.m[w_update], self.M[w_update])-per(self.m[w_update], self.M[w_update]))

                    self.s[w_update] = self.s[w_update] - self.lr * error_w * self.w[w_update] * mjoutput_w[w_update] * (-2) * self.s[w_update] * (perj(x1[i][2], self.m[w_update], self.M[w_update])-per(self.m[w_update], self.M[w_update]))**2

                # h-axis update
                for h_update in range(len(self.middle)):
                    if x1[i][3] > self.M[h_update] : #update M 
                        self.M[h_update] = self.M[h_update] - self.lr * error_h * self.w[h_update] * mjoutput_h[h_update] * (self.s[h_update]**2) * 2 * (perj(x1[i][3], self.m[h_update], self.M[h_update])-per(self.m[h_update], self.M[h_update]))

                    elif x1[i][3] < self.m[h_update] : #update m 
                        self.m[h_update] = self.m[h_update] - self.lr * error_h * self.w[h_update] * mjoutput_h[h_update] * -(self.s[h_update]**2) * 2 * (perj(x1[i][3], self.m[h_update], self.M[h_update])-per(self.m[h_update], self.M[h_update]))

                    self.s[h_update] = self.s[h_update] - self.lr * error_h * self.w[h_update] * mjoutput_h[h_update] * (-2) * self.s[h_update] * (perj(x1[i][3], self.m[h_update], self.M[h_update])-per(self.m[h_update], self.M[h_update]))**2



            # calculate avg loss
            cal_loss = (cal_loss/len(x1))
            cal_error = (cal_error/len(x1))
            cal_error_list = cal_error.numpy().tolist()
            print('Average Loss: {0:.4f}'.format(cal_loss))
            print('Average Error: {0:.4f}'.format(cal_error))
            print(self.w, self.b, self.M, self.m, self.s)
            
            error.append(cal_error_list)
        
        
        
        print(error)
        plot_x = [epoc for epoc in range(1,self.epochs + 1)] 
        plot_y = error
        plt.plot(plot_x, plot_y)
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\model_weight\\error_epoch.png")
        plt.show()
        

        '''
        # save model weights
        temp_cal_loss = cal_loss
        
        
        if temp_cal_loss > cal_loss :
            temp_cal_loss = cal_loss
            self.model_w, self.model_b, self.model_M, self.model_m, self.model_s = self.w, self.b, self.M, self.m, self.s
        else:
            self.w, self.b, self.M, self.m, self.s = self.w, self.b, self.M, self.m, self.s
        #print(temp_cal_loss)
        #print(self.model_w, self.model_b, self.model_M, self.model_m, self.model_s)
        '''

    def predict(self, x1) :
        y_pred = []
        for i in range(len(x1)) : 
            mjoutput_x = torch.FloatTensor([])
            mjoutput_y = torch.FloatTensor([])
            mjoutput_w = torch.FloatTensor([])
            mjoutput_h = torch.FloatTensor([])

            for j in range(len(self.middle)):
            #forward pass
                
                mjoutput_x_temp =  torch.FloatTensor([forward_mj(x1[i][0], self.s[j] , self.m[j] , self.M[j] )])
                mjoutput_x = torch.cat((mjoutput_x, mjoutput_x_temp))

                mjoutput_y_temp =  torch.FloatTensor([forward_mj(x1[i][1], self.s[j] , self.m[j] , self.M[j] )])
                mjoutput_y = torch.cat((mjoutput_y, mjoutput_y_temp))

                mjoutput_w_temp =  torch.FloatTensor([forward_mj(x1[i][2], self.s[j] , self.m[j] , self.M[j] )])
                mjoutput_w = torch.cat((mjoutput_w, mjoutput_w_temp))

                mjoutput_h_temp =  torch.FloatTensor([forward_mj(x1[i][3], self.s[j] , self.m[j] , self.M[j] )])
                mjoutput_h = torch.cat((mjoutput_h, mjoutput_h_temp))
            

            final_output_x = torch.dot(mjoutput_x,self.w) + self.b
            final_output_y = torch.dot(mjoutput_y,self.w) + self.b
            final_output_w = torch.dot(mjoutput_w,self.w) + self.b
            final_output_h = torch.dot(mjoutput_h,self.w) + self.b
            #final_output_concat = torch.cat((final_output_x,final_output_y,final_output_w,final_output_h))

            y_pred.append(final_output_x)
            y_pred.append(final_output_y)
            y_pred.append(final_output_w)
            y_pred.append(final_output_h)
        return np.array(y_pred)

    def save_model_weight(self, ) :
        self.model_M = self.M.numpy()
        self.model_m = self.m.numpy()
        self.model_s = self.s.numpy()
        self.model_w = self.w.numpy()
        self.model_b = self.b.numpy()
        print(self.model_w, '\n', self.model_b,'\n',  self.model_M, '\n',self.model_m,'\n',  self.model_s)
        #model_weight = np.array([self.model_M, self.model_m, self.model_s, self.model_w, self.model_b])
        #np.append(self.model_M, self.model_m, self.model_s, self.model_w, self.model_b)
        np.savetxt('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\model_weight\\model_M.csv', self.model_M, delimiter=',')
        np.savetxt('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\model_weight\\model_mm.csv', self.model_m, delimiter=',')
        np.savetxt('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\model_weight\\model_s.csv', self.model_s, delimiter=',')
        np.savetxt('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\model_weight\\model_w.csv', self.model_w, delimiter=',')
        np.savetxt('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\model_weight\\model_b.csv', self.model_b, delimiter=',')

if __name__ == "__main__":

    #dataset setting
    data_set = np.loadtxt('D:\\Tom Peng\\Electric GoKart\\algorithm\\rbfnet\\test_img\\FHRCNN_dataset1.csv',delimiter= ',')


    print(data_set.shape)

    x1 = torch.tensor(data_set[:,0:4], dtype  = torch.int64)
    
    y = torch.tensor(data_set[:,5:9], dtype  = torch.int64)

    #dataset setting
    

    FHRCNN_RBFNET = FHRCNN_Network(k = 31,lr = 0.01)
    FHRCNN_RBFNET.train(x1,y)

    y_pred = FHRCNN_RBFNET.predict(x1)
    y_pred = np.reshape(y_pred, (len(x1),4))

    print(y_pred)

    y_pred = np.array(y_pred)
    np.savetxt('new_structure_y_pred.csv', y_pred, delimiter=',')

    FHRCNN_RBFNET.save_model_weight()    