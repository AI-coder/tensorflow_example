def focal_loss(logits,labels,alpha=0.25,gamma=2):
    labels = tf.one_hot(labels,2)
    zeros =tf.zeros_like(logits,dtype=logits.dtype)
    pos_error = tf.where(labels>zeros,labels-logits,zeros)
    neg_error = tf.where(labels>zeros,zeros,logits)
    safe_prediction = tf.clip_by_value(logits,1e-10,1- 1e-10)
    fl_loss = - alpha * ( pos_error ** gamma ) * tf.log( safe_prediction)-(1-alpha)*(neg_error**gamma)*tf.log(1.0-safe_prediction)
    return tf.reduce_mean(tf.reduce_sum(fl_loss,-1),-1)