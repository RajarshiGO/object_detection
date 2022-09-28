def custom_loss(y_true, y_pred):
    binary_crossentropy = prob_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    
    prob_loss = binary_crossentropy(
        tf.concat([y_true[:,:,:,0], y_true[:,:,:,5]], axis=0), 
        tf.concat([y_pred[:,:,:,0], y_pred[:,:,:,5]], axis=0)
    )
    
    xy_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:,:,:,1:3], y_true[:,:,:,6:8]], axis=0), 
        tf.concat([y_pred[:,:,:,1:3], y_pred[:,:,:,6:8]], axis=0)
    )
    
    wh_loss = tf.keras.losses.MSE(
        tf.concat([y_true[:,:,:,3:5], y_true[:,:,:,8:10]], axis=0), 
        tf.concat([y_pred[:,:,:,3:5], y_pred[:,:,:,8:10]], axis=0)
    )
    
    bboxes_mask = get_mask(y_true)
    
    xy_loss = xy_loss * bboxes_mask
    wh_loss = wh_loss * bboxes_mask
    
    return prob_loss + xy_loss + wh_loss


def get_mask(y_true):
    anchor_one_mask = tf.where(
        y_true[:,:,:,0] == 0, 
        0.5, 
        5.0
    )
    
    anchor_two_mask = tf.where(
        y_true[:,:,:,5] == 0, 
        0.5, 
        5.0
    )
    
    bboxes_mask = tf.concat(
        [anchor_one_mask,anchor_two_mask],
        axis=0
    )
    
    return bboxes_mask

def process_predictions(predictions, image_grid):
    predictions = prediction_to_bbox(predictions, image_grid)
    bboxes = non_max_suppression(predictions, top_n=100)
        
    # back to coco shape
    bboxes[:,2:4] = bboxes[:,2:4] - bboxes[:,0:2]
    
    return bboxes

def prediction_to_bbox(bboxes, image_grid):    
    bboxes = bboxes.copy()
    
    im_width = (image_grid[:,2] * 32)
    im_height = (image_grid[:,3] * 32)
    
    # descale x,y
    bboxes[:,1] = (bboxes[:,1] * image_grid[:,2]) + image_grid[:,0]
    bboxes[:,2] = (bboxes[:,2] * image_grid[:,3]) + image_grid[:,1]
    bboxes[:,6] = (bboxes[:,6] * image_grid[:,2]) + image_grid[:,0]
    bboxes[:,7] = (bboxes[:,7] * image_grid[:,3]) + image_grid[:,1]
    
    # descale width,height
    bboxes[:,3] = bboxes[:,3] * im_width 
    bboxes[:,4] = bboxes[:,4] * im_height
    bboxes[:,8] = bboxes[:,8] * im_width 
    bboxes[:,9] = bboxes[:,9] * im_height
    
    # centre x,y to top left x,y
    bboxes[:,1] = bboxes[:,1] - (bboxes[:,3] / 2)
    bboxes[:,2] = bboxes[:,2] - (bboxes[:,4] / 2)
    bboxes[:,6] = bboxes[:,6] - (bboxes[:,8] / 2)
    bboxes[:,7] = bboxes[:,7] - (bboxes[:,9] / 2)
    
    # width,heigth to x_max,y_max
    bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
    bboxes[:,4] = bboxes[:,2] + bboxes[:,4]
    bboxes[:,8] = bboxes[:,6] + bboxes[:,8]
    bboxes[:,9] = bboxes[:,7] + bboxes[:,9]
    
    return bboxes

